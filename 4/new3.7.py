import time

import numpy as np
from collections import deque
from pyomyo import Myo, emg_mode
import serial

# ===== Arduino設定 =====
SERIAL_MOTOR = "COM5"  # モータ制御Arduino
BAUDRATE = 115200

ser_motor = serial.Serial(SERIAL_MOTOR, BAUDRATE, timeout=0.1)

# ===== EMGパラメータ =====
FS = 200.0
RMS_WIN_MS = 80
RMS_WIN = max(5, int(FS * RMS_WIN_MS / 1000.0))
EMA_ALPHA = 0.2
K_SIGMA = 5.6
REFRACTORY_MS = 200

CALIB_DURATION = 3.0


# ===== 状態管理 =====
rms_buf = [deque(maxlen=RMS_WIN) for _ in range(8)]
ema_val = np.zeros(8)
mean = np.zeros(8)
std = np.ones(8)
last_trigger_time = 0
trigger_time = None
direction = None
last_direction = None
arrival_queue = deque(maxlen=RMS_WIN)
up_ch = [1, 2]
down_ch = [5, 6]
calibration_done = False
expected_direction = "UP"


# ===== EMG処理関数 =====
def compute_rms(buf):
    if len(buf) == 0:
        return 0
    arr = np.array(buf)
    return np.sqrt(np.mean(arr**2))


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
        mean[ch] = float(np.mean(arr))
        std[ch] = float(np.std(arr, ddof=1))
        if std[ch] < 1e-6 or np.isnan(std[ch]):
            std[ch] = 1e-6
    calibration_done = True
    print("=== キャリブレーション完了 ===")
    for ch in range(8):
        print(f"ch{ch+1}: mean={mean[ch]:.6f} std={std[ch]:.6f}")


# ===== サーボ制御 =====
def send_motor_command(direction):
    if direction == "UP":
        ser_motor.write(b"MOTOR+\n")
    elif direction == "DOWN":
        ser_motor.write(b"MOTOR-\n")


# ===== EMGハンドラ =====
def on_emg(emg, movement):
    global calibration_done, expected_direction
    global ema_val, mean, std
    global last_trigger_time, direction, last_direction
    global arrival_queue

    if emg is None:
        return
    t_arrival = time.time()
    arrival_queue.append(t_arrival)

    # RMS + EMA
    rms_vals = []
    for ch in range(8):
        rms_buf[ch].append(emg[ch])
        rms = compute_rms(rms_buf[ch])
        rms_vals.append(rms)
        ema_val[ch] = EMA_ALPHA * rms + (1 - EMA_ALPHA) * ema_val[ch]

    if not calibration_done:
        return

    z_scores = (ema_val - mean) / (std + 1e-6)

    if t_arrival - last_trigger_time > REFRACTORY_MS / 1000.0:
        for ch in up_ch + down_ch:
            if z_scores[ch] > K_SIGMA:
                new_direction = "UP" if ch in up_ch else "DOWN"
                if last_direction is None and new_direction != "UP":
                    continue
                if new_direction != expected_direction:
                    continue

                # 即時実行
                send_motor_command(new_direction)
                print(f"[TRIGGER] ch={ch} dir={new_direction} z={z_scores[ch]:.2f}")

                last_trigger_time = t_arrival
                last_direction = new_direction
                expected_direction = "DOWN" if new_direction == "UP" else "UP"
                break


# ===== Myo初期化 =====
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

    # ファイルを確実に閉じる（bare except禁止）
    # シリアルを確実に閉じる
    for s, name in [(ser_motor, "ser_motor")]:
        try:
            s.close()
        except Exception as e:
            print(f"[WARN] {name} を閉じる際にエラー: {e}")

    # Myo切断
    try:
        m.disconnect()
    except Exception as e:
        print(f"[WARN] Myo切断時にエラー: {e}")

    print("[INFO] 安全に終了しました")
