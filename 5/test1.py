import time
import csv
import numpy as np
from collections import deque
from pyomyo import Myo, emg_mode
import serial
import threading

# ===== 設定 =====
SERIAL_MOTOR = "COM4"
SERIAL_SENSOR = "COM5"
BAUDRATE = 115200

FS = 200.0
RMS_WIN_MS = 50
RMS_WIN = max(7, int(FS * RMS_WIN_MS / 1000.0))
EMA_ALPHA = 0.2
K_SIGMA = 6.0
REFRACTORY_MS = 200
PEAK_WIN_MS = 100
CALIB_DURATION = 3.0

CSV_PATH = "acc_delay_log_safe2.csv"
RAW_CSV_PATH = "emg_raw_safe2.csv"
RMS_CSV_PATH = "emg_rms_safe2.csv"
EMA_CSV_PATH = "emg_ema_safe2.csv"

MAX_SENSOR_WAIT = 1.0


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

# ===== CSV初期化 =====
csv_file = open(CSV_PATH, "w", newline="")
raw_csv_file = open(RAW_CSV_PATH, "w", newline="")
rms_csv_file = open(RMS_CSV_PATH, "w", newline="")
ema_csv_file = open(EMA_CSV_PATH, "w", newline="")

csv_writer = csv.writer(csv_file)
raw_csv_writer = csv.writer(raw_csv_file)
rms_csv_writer = csv.writer(rms_csv_file)
ema_csv_writer = csv.writer(ema_csv_file)

csv_writer.writerow(
    [
        "utc_ts",
        "local_time",
        "sensor1_time",
        "sensor2_time",
        "total_delay_ms",
        "proc_latency_ms",
        "motor_latency_ms",
    ]
)
raw_csv_writer.writerow(["utc_ts", "local_time"] + [f"CH{i+1}" for i in range(8)])
rms_csv_writer.writerow(["utc_ts", "local_time"] + [f"CH{i+1}" for i in range(8)])
ema_csv_writer.writerow(["utc_ts", "local_time"] + [f"CH{i+1}" for i in range(8)])

# ===== 状態 =====
rms_buf = [deque(maxlen=RMS_WIN) for _ in range(8)]
ema_val = np.zeros(8)
mean = np.zeros(8)
std = np.ones(8)
last_trigger_time = 0
trigger_time = None
trigger_ch = None
direction = None
last_direction = None
peak_val = -np.inf
peak_time = None
arrival_queue = deque(maxlen=RMS_WIN)
arrival_at_trigger = None
up_ch = [0, 1]
down_ch = [4, 5]
calibration_done = False
expected_direction = "UP"

proc_latency_ms = None
motor_latency_ms = None


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


# ===== キャリブレーション =====
def calibrate(duration=CALIB_DURATION):
    global mean, std, calibration_done
    print("=== キャリブレーション開始: 3秒間安静してください ===")
    cal_buf = [[] for _ in range(8)]
    start_time = time.time()
    while time.time() - start_time < duration:
        m.run()
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
    print("=== キャリブレーション完了 ===")
    for ch in range(8):
        print(f"ch{ch+1}: mean={mean[ch]:.6f} std={std[ch]:.6f}")


# ===== モータ制御 =====
def send_motor_command(direction):
    if ser_motor is None:
        print("[WARN] ser_motor not opened; skipping motor command")
        return
    try:
        if direction == "UP":
            ser_motor.write(b"MOTOR+\n")
        elif direction == "DOWN":
            ser_motor.write(b"MOTOR-\n")
        ser_motor.flush()
    except Exception as e:
        print(f"[ERROR] sending motor command failed: {e}")


# ===== センサ遅延計測（スレッド用） =====
def measure_sensor_delay_async(proc_latency_ms, motor_latency_ms, dir_at_trigger):
    if ser_sensor is None:
        return
    acc1_delta = None
    acc2_delta = None
    start = time.time()
    while time.time() - start < MAX_SENSOR_WAIT:
        try:
            line_bytes = ser_sensor.readline()
        except Exception:
            break
        if not line_bytes:
            time.sleep(0.005)
            continue
        line = line_bytes.decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        try:
            if line.startswith("ACC1:"):
                acc1_delta = float(line.split(":", 1)[1])
            elif line.startswith("ACC2:"):
                acc2_delta = float(line.split(":", 1)[1])
        except:
            continue
        if acc1_delta is not None and acc2_delta is not None:
            break

    if acc1_delta is not None and acc2_delta is not None:
        total_delay = abs(acc2_delta - acc1_delta)
        utc_ts, local = utc_and_local()
        print(
            f"[ACC_DELAY] total={total_delay:.2f} ms (proc={proc_latency_ms:.2f}, motor={motor_latency_ms:.2f})"
        )
        csv_writer.writerow(
            [
                utc_ts,
                local,
                acc1_delta,
                acc2_delta,
                total_delay,
                proc_latency_ms,
                motor_latency_ms,
            ]
        )
        csv_file.flush()


# ===== EMGハンドラ =====
def on_emg(emg, movement):
    global calibration_done, expected_direction
    global ema_val, mean, std
    global last_trigger_time, trigger_time, trigger_ch, direction, last_direction
    global peak_val, peak_time, arrival_queue, arrival_at_trigger
    global proc_latency_ms, motor_latency_ms

    if emg is None:
        return
    t_arrival = time.time()
    arrival_queue.append(t_arrival)

    # RAW保存
    utc_ts, local = utc_and_local()
    raw_csv_writer.writerow([utc_ts, local] + list(emg))
    raw_csv_file.flush()

    # RMS + EMA 更新
    rms_vals = []
    for ch in range(8):
        rms_buf[ch].append(float(emg[ch]))
        rms = compute_rms(rms_buf[ch])
        rms_vals.append(rms)
        ema_val[ch] = EMA_ALPHA * rms + (1 - EMA_ALPHA) * ema_val[ch]
    rms_csv_writer.writerow([utc_ts, local] + list(rms_vals))
    rms_csv_file.flush()
    ema_csv_writer.writerow([utc_ts, local] + list(ema_val))
    ema_csv_file.flush()

    if not calibration_done:
        return

    z_scores = (ema_val - mean) / (std + 1e-6)

    # トリガ検出
    if trigger_time is None:
        if t_arrival - last_trigger_time > REFRACTORY_MS / 1000.0:
            for ch in up_ch + down_ch:
                if z_scores[ch] > K_SIGMA:
                    new_direction = "UP" if ch in up_ch else "DOWN"
                    if last_direction is None and new_direction != "UP":
                        continue
                    if new_direction != expected_direction:
                        continue
                    trigger_time = time.time()
                    trigger_ch = ch
                    direction = new_direction
                    peak_val = z_scores[ch]
                    peak_time = trigger_time
                    arrival_at_trigger = (
                        arrival_queue[0] if len(arrival_queue) > 0 else t_arrival
                    )
                    proc_latency_ms = (trigger_time - arrival_at_trigger) * 1000.0
                    last_trigger_time = trigger_time
                    print(
                        f"[TRIGGER] ch={ch+1} dir={direction} z={z_scores[ch]:.2f} proc_latency={proc_latency_ms:.2f}ms"
                    )
                    break
    else:
        # ピーク更新
        if z_scores[trigger_ch] > peak_val:
            peak_val = z_scores[trigger_ch]
            peak_time = t_arrival
        # PEAK_WIN_MS経過で即時モータ + センサ計測スレッド
        if (t_arrival - trigger_time) * 1000.0 > PEAK_WIN_MS:
            motor_latency_ms = (time.time() - trigger_time) * 1000.0
            # 1. 義手即動作
            send_motor_command(direction)
            # 2. センサ計測スレッド起動
            threading.Thread(
                target=measure_sensor_delay_async,
                args=(proc_latency_ms, motor_latency_ms, direction),
            ).start()
            last_direction = direction
            expected_direction = "DOWN" if direction == "UP" else "UP"
            # トリガリセット
            trigger_time = None
            trigger_ch = None
            direction = None
            peak_val = -np.inf
            peak_time = None
            arrival_at_trigger = None
            arrival_queue.clear()


# ===== Myo 初期化 =====
m = Myo(mode=emg_mode.RAW)
m.connect()
m.add_emg_handler(on_emg)
m.vibrate(1)

# キャリブレーション
calibrate()

print("==== START ====")
try:
    while True:
        m.run()
except KeyboardInterrupt:
    print("\n[INFO] Ctrl+C で終了しました")
finally:
    print("[INFO] リソースを閉じています...")
    for f, name in [
        (csv_file, "csv_file"),
        (raw_csv_file, "raw_csv_file"),
        (rms_csv_file, "rms_csv_file"),
        (ema_csv_file, "ema_csv_file"),
    ]:
        try:
            f.close()
        except Exception as e:
            print(f"[WARN] {name} を閉じる際にエラー: {e}")

    for s, name in [(ser_motor, "ser_motor"), (ser_sensor, "ser_sensor")]:
        try:
            if s is not None:
                s.close()
        except Exception as e:
            print(f"[WARN] {name} を閉じる際にエラー: {e}")

    try:
        m.disconnect()
    except Exception as e:
        print(f"[WARN] Myo切断時にエラー: {e}")

    print("[INFO] 安全に終了しました")
