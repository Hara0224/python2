import time
import csv

# import sys
import numpy as np
from collections import deque
from pyomyo import Myo, emg_mode
import serial

# ===== Arduino設定 =====
SERIAL_PORT = "COM5"
BAUDRATE = 115200
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)

# ===== パラメータ =====
FS = 200.0
RMS_WIN_MS = 20
RMS_WIN = max(3, int(FS * RMS_WIN_MS / 1000.0))
EMA_ALPHA = 0.2
K_SIGMA = 4.0
REFRACTORY_MS = 120
PEAK_WIN_MS = 80
MAX_MOTOR_WAIT = 0.2  # 500ms
CALIB_DURATION = 3.0  # キャリブ時間[s]

# ===== CSV保存 =====
CSV_PATH = "emg_delay_log.csv"
csv_file = open(CSV_PATH, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(
    [
        "trigger_ch",
        "direction",
        "trigger_time",
        "hit_time",
        "sw_delay_ms",
        "pc_delay_ms",
        "motor_delay_ms",
        "total_delay_ms",
    ]
)

# ===== 状態管理 =====
rms_buf = [deque(maxlen=RMS_WIN) for _ in range(8)]
ema_val = np.zeros(8)
mean = np.zeros(8)
std = np.ones(8)

last_trigger_time = 0
trigger_time = None
trigger_ch = None
direction = None
peak_val = -np.inf
peak_time = None

# チャンネル設定
up_ch = [1, 2]  # +方向
down_ch = [5, 6]  # -方向


def compute_rms(buf):
    if len(buf) == 0:
        return 0
    arr = np.array(buf)
    return np.sqrt(np.mean(arr**2))


def calibrate(m):
    """最初に3秒間安静状態を記録して平均・標準偏差を計算"""
    print("=== キャリブレーション開始: 3秒間安静してください ===")
    start = time.time()
    cal_buf = [[] for _ in range(8)]
    while time.time() - start < CALIB_DURATION:
        m.run()
        for ch in range(8):
            cal_buf[ch].append(compute_rms(rms_buf[ch]) if rms_buf[ch] else 0)
        time.sleep(0.001)

    global mean, std
    for ch in range(8):
        arr = np.array(cal_buf[ch])
        mean[ch] = np.mean(arr)
        std[ch] = np.std(arr, ddof=1) + 1e-6

    print("=== キャリブレーション完了 ===")
    for ch in range(8):
        print(f"ch{ch+1}: mean={mean[ch]:.3f} std={std[ch]:.3f}")


def on_emg(emg, movement):
    global ema_val, mean, std
    global last_trigger_time, trigger_time, trigger_ch, direction
    global peak_val, peak_time

    t_now = time.time()

    # ---- RMS + EMA 更新 ----
    for ch in range(8):
        rms_buf[ch].append(emg[ch])
        rms = compute_rms(rms_buf[ch])
        ema_val[ch] = EMA_ALPHA * rms + (1 - EMA_ALPHA) * ema_val[ch]

    z_scores = (ema_val - mean) / (std + 1e-6)

    # ---- トリガー判定 ----
    if trigger_time is None:
        if t_now - last_trigger_time > REFRACTORY_MS / 1000.0:
            for ch in up_ch + down_ch:
                if z_scores[ch] > K_SIGMA:
                    trigger_time = t_now
                    trigger_ch = ch
                    direction = "UP" if ch in up_ch else "DOWN"
                    peak_val = z_scores[ch]
                    peak_time = t_now
                    last_trigger_time = t_now
                    print(f"[TRIGGER] ch={ch} dir={direction} z={z_scores[ch]:.2f}")
                    break
    else:
        # ---- PEAK検出 ----
        if z_scores[trigger_ch] > peak_val:
            peak_val = z_scores[trigger_ch]
            peak_time = t_now

        if (t_now - trigger_time) * 1000.0 > PEAK_WIN_MS:
            # ---- HIT確定 ----
            send_motor_command(direction, trigger_time)
            trigger_time = None
            trigger_ch = None
            direction = None
            peak_val = -np.inf
            peak_time = None


def send_motor_command(direction, t_trigger):
    cmd = "MOTOR+" if direction == "UP" else "MOTOR-"
    ser.write((cmd + "\n").encode())

    t_send = time.time()
    hit_time = None
    while time.time() - t_send < MAX_MOTOR_WAIT:
        line = ser.readline().decode().strip()
        if line.startswith("HIT"):
            hit_time = time.time()
            break

    if hit_time:
        sw_delay = (t_trigger - last_trigger_time) * 1000
        pc_delay = (t_send - t_trigger) * 1000
        motor_delay = (hit_time - t_send) * 1000
        total_delay = (hit_time - t_trigger) * 1000

        print(f"[HIT] {direction} sw={sw_delay:.1f}ms pc={pc_delay:.1f}ms " f"motor={motor_delay:.1f}ms total={total_delay:.1f}ms")

        csv_writer.writerow(
            [
                trigger_ch,
                direction,
                t_trigger,
                hit_time,
                sw_delay,
                pc_delay,
                motor_delay,
                total_delay,
            ]
        )
        csv_file.flush()
    else:
        print("[TIMEOUT] Arduino応答なし")


# ===== Myo 初期化 =====
m = Myo(mode=emg_mode.RAW)
m.connect()
m.add_emg_handler(on_emg)
m.vibrate(1)

calibrate(m)

print("==== START ====")
try:
    while True:
        m.run()
except KeyboardInterrupt:
    pass
finally:
    csv_file.close()
    ser.close()
    m.disconnect()
