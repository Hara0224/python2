import time
import csv
import numpy as np
from collections import deque
from pyomyo import Myo, emg_mode
import serial

# ===== Arduino設定 =====
SERIAL_PORT = "COM5"  # 可変
BAUDRATE = 115200
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)

# ===== パラメータ =====
FS = 200.0
RMS_WIN_MS = 50  # 20であると少ないのでは無いかという疑問があったので，適宜増やしてみる(または減らす)
RMS_WIN = max(3, int(FS * RMS_WIN_MS / 1000.0))
EMA_ALPHA = 0.2
K_SIGMA = 5.5
REFRACTORY_MS = 200
PEAK_WIN_MS = 100
MAX_MOTOR_WAIT = 0.2
CALIB_DURATION = 3.0  # キャリブ時間[s]

# ===== CSV保存 =====
CSV_PATH = "emg_delay_log_sw2.csv"
RAW_CSV_PATH = "emg_raw.csv"
RMS_CSV_PATH = "emg_rms.csv"
EMA_CSV_PATH = "emg_ema.csv"

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
raw_csv_writer.writerow(["timestamp"] + [f"ch{i+1}" for i in range(8)])
rms_csv_writer.writerow(["timestamp"] + [f"ch{i+1}" for i in range(8)])
ema_csv_writer.writerow(["timestamp"] + [f"ch{i+1}" for i in range(8)])

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
last_direction = None

# チャンネル設定
up_ch = [1, 2]
down_ch = [5, 6]

# トリガ判定用の到着時刻キュー
arrival_queue = deque(maxlen=RMS_WIN)
arrival_at_trigger = None


def compute_rms(buf):
    if len(buf) == 0:
        return 0
    arr = np.array(buf)
    return np.sqrt(np.mean(arr**2))


def calibrate(m):
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
    global last_trigger_time, trigger_time, trigger_ch, direction, last_direction
    global peak_val, peak_time, arrival_queue, arrival_at_trigger

    t_arrival = time.time()
    arrival_queue.append(t_arrival)  # 到着時刻をキューに追加

    # ---- RAW値保存 ----
    raw_csv_writer.writerow([t_arrival] + list(emg))
    raw_csv_file.flush()

    # ---- RMS + EMA 更新 ----
    rms_vals = []
    for ch in range(8):
        rms_buf[ch].append(emg[ch])
        rms = compute_rms(rms_buf[ch])
        rms_vals.append(rms)
        ema_val[ch] = EMA_ALPHA * rms + (1 - EMA_ALPHA) * ema_val[ch]

    # ---- RMS値保存 ----
    rms_csv_writer.writerow([t_arrival] + list(rms_vals))
    rms_csv_file.flush()

    # ---- EMA値保存 ----
    ema_csv_writer.writerow([t_arrival] + list(ema_val))
    ema_csv_file.flush()

    z_scores = (ema_val - mean) / (std + 1e-6)

    # ---- トリガ判定 ----
    if trigger_time is None:
        if t_arrival - last_trigger_time > REFRACTORY_MS / 1000.0:
            for ch in up_ch + down_ch:
                if z_scores[ch] > K_SIGMA:
                    new_direction = "UP" if ch in up_ch else "DOWN"
                    if new_direction != last_direction:
                        trigger_time = time.time()  # 判定時刻
                        trigger_ch = ch
                        direction = new_direction
                        peak_val = z_scores[ch]
                        peak_time = trigger_time
                        arrival_at_trigger = arrival_queue[0]  # キューの最初の到着時刻
                        last_trigger_time = trigger_time
                        print(f"[TRIGGER] ch={ch} dir={direction} z={z_scores[ch]:.2f}")
                    break
    else:
        # ---- PEAK検出 ----
        if z_scores[trigger_ch] > peak_val:
            peak_val = z_scores[trigger_ch]
            peak_time = t_arrival

        if (t_arrival - trigger_time) * 1000.0 > PEAK_WIN_MS:
            send_motor_command(direction, trigger_time, arrival_at_trigger)
            last_direction = direction
            trigger_time = None
            trigger_ch = None
            direction = None
            peak_val = -np.inf
            peak_time = None
            arrival_at_trigger = None


def send_motor_command(direction, trigger_time, arrival_time):
    cmd = "MOTOR+" if direction == "UP" else "MOTOR-"
    t_send = time.time()
    ser.write((cmd + "\n").encode())

    hit_time = None
    start = time.time()
    while time.time() - start < MAX_MOTOR_WAIT:
        line = ser.readline().decode().strip()
        if line.startswith("HIT"):
            hit_time = time.time()
            break

    if hit_time:
        sw_delay = (trigger_time - arrival_time) * 1000  # 生データ到着〜トリガ判定
        pc_delay = (t_send - trigger_time) * 1000  # トリガ判定〜送信
        motor_delay = (hit_time - t_send) * 1000
        total_delay = (hit_time - arrival_time) * 1000

        print(
            f"[HIT] {direction} sw={sw_delay:.1f}ms pc={pc_delay:.1f}ms "
            f"motor={motor_delay:.1f}ms total={total_delay:.1f}ms"
        )

        csv_writer.writerow(
            [
                trigger_ch,
                direction,
                trigger_time,
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
    print("\n[INFO] Ctrl+C で終了しました")
finally:
    print("[INFO] リソースを閉じています...")
    try:
        csv_file.close()
    except Exception:
        pass
    try:
        raw_csv_file.close()
        rms_csv_file.close()
        ema_csv_file.close()
    except Exception:
        pass
    try:
        ser.close()
    except Exception:
        pass
    try:
        m.disconnect()
    except Exception:
        pass
    print("[INFO] 安全に終了しました")
