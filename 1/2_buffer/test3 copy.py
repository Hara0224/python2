import time
import csv
import numpy as np
from collections import deque
from pyomyo import Myo, emg_mode
import serial
import threading

# ===== 設定 =====
SERIAL_MOTOR = "COM5"
SERIAL_SENSOR = "COM4"
BAUDRATE = 115200

FS = 200.0
RMS_WIN_MS = 100
RMS_WIN = max(5, int(FS * RMS_WIN_MS / 1000.0))
EMA_ALPHA = 0.2
K_SIGMA = 7.0
REFRACTORY_MS = 150
PEAK_WIN_MS = 100
CALIB_DURATION = 3.0
MAX_SENSOR_WAIT = 1.0
PRE_TRIGGER_MARGIN = 1.5
SLOPE_MIN = 0.5

UP_CH = [1, 2]  # ch2,ch3
DOWN_CH = [4, 5]  # ch5,ch6

OLD_DELAY_MS = 160.0  # 旧処理遅延

# CSVログ
CSV_PATH = "trigger_log_motor1.csv"
csv_file = open(CSV_PATH, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(
    [
        "timestamp",
        "trigger_type",
        "delay_old_ms",
        "delay_new_ms",
        "delay_reduction_ms",
        "z_score",
        "slope",
        "rms",
        "acc1_time",
        "acc2_time",
        "total_sensor_delay_ms",
    ]
)


# ===== シリアルオープン =====
def safe_serial_open(port, baud, timeout=0.1):
    try:
        s = serial.Serial(port, baud, timeout=timeout)
        print(f"[INFO] Opened serial {port} @ {baud}")
        return s
    except Exception as e:
        print(f"[ERROR] Cannot open serial {port}: {e}")
        return None


ser_motor = safe_serial_open(SERIAL_MOTOR, BAUDRATE)
ser_sensor = safe_serial_open(SERIAL_SENSOR, BAUDRATE)

# ===== 状態 =====
rms_buf = [deque(maxlen=RMS_WIN) for _ in range(8)]
ema_val = np.zeros(8)
prev_ema = np.zeros(8)
mean = np.zeros(8)
std = np.ones(8)
last_trigger_time = 0
expected_direction = "UP"
calibration_done = False


# ===== ユーティリティ =====
def utc_and_local():
    utc_ts = time.time()
    local = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(utc_ts))
    return utc_ts, local


def compute_rms(buf):
    if len(buf) == 0:
        return 0.0
    arr = np.array(buf, dtype=float)
    return float(np.sqrt(np.mean(arr**2)))


def send_motor_command(direction):
    if ser_motor is None:
        return
    try:
        cmd = "MOTOR+\n" if direction == "UP" else "MOTOR-\n"
        ser_motor.write(cmd.encode())
        ser_motor.flush()
        print(f"[MOTOR] Sent command: {cmd.strip()}")
    except Exception as e:
        print(f"[ERROR] Motor command failed: {e}")


def measure_sensor_delay():
    if ser_sensor is None:
        return None, None, None
    acc1_time = None
    acc2_time = None
    start_time = time.time()
    while time.time() - start_time < MAX_SENSOR_WAIT:
        try:
            line_bytes = ser_sensor.readline()
        except Exception:
            break
        if not line_bytes:
            time.sleep(0.001)
            continue
        try:
            line = line_bytes.decode("utf-8").strip()
            if line.startswith("ACC1:"):
                acc1_time = float(line.split(":")[1])
            elif line.startswith("ACC2:"):
                acc2_time = float(line.split(":")[1])
        except:
            continue
        if acc1_time is not None and acc2_time is not None:
            break
    if acc1_time is not None and acc2_time is not None:
        total_sensor_delay = abs(acc2_time - acc1_time)
    else:
        total_sensor_delay = None
    return acc1_time, acc2_time, total_sensor_delay


# ===== トリガ判定 =====
def detect_trigger(ch_idx, z, slope):
    global last_trigger_time, expected_direction
    t_now = int(time.time() * 1000)
    if t_now - last_trigger_time < REFRACTORY_MS:
        return None
    dir_now = "UP" if ch_idx in UP_CH else "DOWN"
    if expected_direction != dir_now:
        return None
    trigger_type = None
    if z > K_SIGMA:
        trigger_type = "NORMAL"
    elif z > (K_SIGMA - PRE_TRIGGER_MARGIN) and slope > SLOPE_MIN:
        trigger_type = "EARLY"
    if trigger_type:
        last_trigger_time = t_now
        expected_direction = "DOWN" if dir_now == "UP" else "UP"
    return trigger_type


# ===== EMGコールバック =====
def on_emg(emg, moving):
    global ema_val, prev_ema
    rms_vals = []
    for ch in range(8):
        rms_buf[ch].append(abs(emg[ch]))
        rms = compute_rms(rms_buf[ch])
        rms_vals.append(rms)
        ema_val[ch] = EMA_ALPHA * rms + (1 - EMA_ALPHA) * ema_val[ch]
    if not calibration_done:
        return
    z_scores = (ema_val - mean) / (std + 1e-6)
    slopes = ema_val - prev_ema
    prev_ema[:] = ema_val[:]

    for ch in UP_CH + DOWN_CH:
        z = z_scores[ch]
        slope = slopes[ch]
        trigger = detect_trigger(ch, z, slope)
        if trigger:
            send_motor_command("UP" if ch in UP_CH else "DOWN")
            acc1, acc2, total_sensor_delay = measure_sensor_delay()
            delay_new = 0.0
            delay_reduction = OLD_DELAY_MS - delay_new
            utc_ts, _ = utc_and_local()
            csv_writer.writerow(
                [
                    utc_ts,
                    trigger,
                    OLD_DELAY_MS,
                    delay_new,
                    delay_reduction,
                    round(z, 2),
                    round(slope, 3),
                    round(rms_vals[ch], 2),
                    acc1,
                    acc2,
                    total_sensor_delay,
                ]
            )
            csv_file.flush()
            print(
                f"[{trigger}] ch{ch+1} z={z:.2f} slope={slope:.3f} rms={rms_vals[ch]:.2f} sensor_delay={total_sensor_delay}"
            )


# ===== キャリブレーション =====
def calibrate(myo, duration=CALIB_DURATION):
    global mean, std, calibration_done, ema_val
    print("=== キャリブレーション開始 ===")
    cal_buf = [[] for _ in range(8)]
    start_time = time.time()
    while time.time() - start_time < duration:
        myo.run()
        time.sleep(0.001)
        for ch in range(8):
            if len(rms_buf[ch]) > 0:
                cal_buf[ch].append(compute_rms(rms_buf[ch]))
    for ch in range(8):
        arr = np.array(cal_buf[ch]) if len(cal_buf[ch]) > 0 else np.array([0.0])
        std_ch = (
            float(np.std(arr, ddof=1)) if arr.size > 1 else float(np.std(arr, ddof=0))
        )
        if std_ch < 1e-6 or np.isnan(std_ch):
            std_ch = 1e-6
        mean[ch] = float(np.mean(arr))
        std[ch] = std_ch
    calibration_done = True
    ema_val[:] = mean[:]
    print("=== キャリブレーション完了 ===")
    for ch in range(8):
        print(f"ch{ch+1}: mean={mean[ch]:.6f}, std={std[ch]:.6f}")


# ===== 実行 =====
m = Myo(mode=emg_mode.RAW)
m.add_emg_handler(on_emg)
m.connect()
print("[INFO] Myo接続完了")

calibrate(m)

print("==== START ====")
try:
    while True:
        m.run()
except KeyboardInterrupt:
    print("\n[INFO] 停止しました")
finally:
    try:
        csv_file.close()
    except:
        pass
    try:
        if ser_motor is not None:
            ser_motor.close()
        if ser_sensor is not None:
            ser_sensor.close()
    except:
        pass
    try:
        m.disconnect()
    except:
        pass
    print("[INFO] 終了しました")
