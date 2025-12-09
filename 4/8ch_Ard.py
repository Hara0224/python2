import time
import csv
import sys
import numpy as np
from pyomyo import Myo, emg_mode
import serial

# ====== Arduino ======
SERIAL_PORT = "COM4"
BAUDRATE = 115200
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)

# ====== 設定 ======
SAVE_CSV = True
CSV_PATH = "emg_stream_log_ch3ch6.csv"

FS = 200.0
DT = 1.0 / FS
RMS_WIN_MS = 20
RMS_WIN = max(3, int(FS * RMS_WIN_MS / 1000.0))
EMA_ALPHA = 0.2
K_SIGMA = 4.0
SLOPE_THR_FACTOR = 0.5
REFRACTORY_MS = 120
PEAK_WIN_MS = 80

# ====== グローバル ======
running = True
calibrated = False

baseline_samples_ch3 = []
baseline_samples_ch6 = []
baseline_duration = 3.0

mu3 = 0.0
sigma3 = 1.0
mu6 = 0.0
sigma6 = 1.0
Smax_cal6 = None

buf = np.zeros((RMS_WIN, 8), dtype=np.float32)
buf_ptr = 0
buf_filled = 0
sumsq = np.zeros(8, dtype=np.float32)

ema_initialized = False
last_hit_time = 0.0
track_peak = False
track_until = 0.0
peak_S6 = 0.0
pending_hit_time = None

# CSV
csv_file = open(CSV_PATH, "w", newline="") if SAVE_CSV else None
csv_writer = csv.writer(csv_file) if csv_file else None
if csv_writer:
    csv_writer.writerow(["Timestamp"] + [f"CH{i+1}" for i in range(8)])


def sliding_rms_update(x):
    global buf, buf_ptr, buf_filled, sumsq
    old = buf[buf_ptr]
    sumsq += x * x - old * old
    buf[buf_ptr] = x
    buf_ptr = (buf_ptr + 1) % RMS_WIN
    buf_filled = min(buf_filled + 1, RMS_WIN)
    if buf_filled < RMS_WIN:
        return None
    return np.sqrt(sumsq / RMS_WIN)


def update_ema(value):
    global ema_initialized
    if not ema_initialized:
        ema_initialized = True
        return value
    else:
        return EMA_ALPHA * value + (1.0 - EMA_ALPHA) * value


def on_emg(emg, movement):
    global calibrated, mu3, sigma3, mu6, sigma6, Smax_cal6
    global last_hit_time, track_peak, track_until, peak_S6, pending_hit_time

    ts = time.time()
    x = np.asarray(emg, dtype=np.float32)

    if csv_writer:
        csv_writer.writerow([ts] + list(map(int, emg)))

    rms_vec = sliding_rms_update(x)
    if rms_vec is None:
        return

    rms_ch3 = rms_vec[2]
    rms_ch6 = rms_vec[5]

    # ===== キャリブ中 =====
    if not calibrated:
        baseline_samples_ch3.append(rms_ch3)
        baseline_samples_ch6.append(rms_ch6)
        return

    # ===== HITトリガー & ピーク追跡 =====
    T3 = mu3 + K_SIGMA * sigma3
    T6 = mu6 + K_SIGMA * sigma6
    now = ts
    slope3 = rms_ch3 - T3

    # ピーク追跡中
    if track_peak:
        if rms_ch6 > peak_S6:
            peak_S6 = rms_ch6
        if now >= track_until:
            # ch6専用 Smax更新
            if Smax_cal6 is None:
                Smax_cal6 = peak_S6
            else:
                Smax_cal6 = max(Smax_cal6 * 0.995, Smax_cal6)
                Smax_cal6 = max(Smax_cal6, peak_S6)

            # 強度0-100%
            lo = T6
            hi = max(T6 + 1e-6, Smax_cal6)
            strength = int(np.clip((peak_S6 - lo) / (hi - lo) * 100.0, 0, 100))

            hit_time = pending_hit_time if pending_hit_time is not None else now
            print(f"[HIT] t={hit_time:.3f} strength={strength:3d} ch6_peak={peak_S6:.3f} T6={T6:.3f}")

            # Arduino送信
            try:
                ser.write(f"HIT,{strength},{hit_time:.6f}\n".encode())
            except Exception as e:
                print("Arduino送信エラー:", e)

            track_peak = False
            last_hit_time = now
            pending_hit_time = None
        return

    # 新しいHIT条件 (ch3基準)
    if (now - last_hit_time) * 1000.0 < REFRACTORY_MS:
        return

    if (rms_ch3 >= T3) and (slope3 >= SLOPE_THR_FACTOR * sigma3):
        pending_hit_time = now
        track_peak = True
        track_until = now + PEAK_WIN_MS / 1000.0
        peak_S6 = rms_ch6


def calibrate_and_run():
    global calibrated, mu3, sigma3, mu6, sigma6

    m = Myo(mode=emg_mode.RAW)
    m.connect()
    m.add_emg_handler(on_emg)

    print("=== キャリブレーション開始 ===")
    print(f"・腕をリラックスして静止してください（約{baseline_duration}秒）...")
    t0 = time.time()
    while (time.time() - t0) < baseline_duration:
        m.run()

    # キャリブ計算
    arr3 = np.asarray(baseline_samples_ch3, dtype=np.float32)
    arr6 = np.asarray(baseline_samples_ch6, dtype=np.float32)
    if arr3.size < 10 or arr6.size < 10:
        print("キャリブ用データが不足")
        sys.exit(1)

    mu3 = float(np.mean(arr3))
    sigma3 = float(np.std(arr3, ddof=1) + 1e-6)
    mu6 = float(np.mean(arr6))
    sigma6 = float(np.std(arr6, ddof=1) + 1e-6)

    print(f"ch3基線: mu3={mu3:.3f}, sigma3={sigma3:.3f}")
    print(f"ch6基線: mu6={mu6:.3f}, sigma6={sigma6:.3f}")
    print(f"しきい値 ch3: {mu3 + K_SIGMA*sigma3:.3f} , ch6: {mu6 + K_SIGMA*sigma6:.3f}")

    calibrated = True
    print("=== 検出開始 === (Ctrl+Cで終了) ===")

    try:
        while True:
            m.run()
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        if csv_file:
            csv_file.close()
        m.disconnect()
        ser.close()
        print("終了")


if __name__ == "__main__":
    calibrate_and_run()
