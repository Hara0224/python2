import time
import csv
import numpy as np
from pyomyo import Myo, emg_mode
import serial
import threading
from datetime import datetime

# ====== Arduino ======
SERIAL_PORT = "COM4"
BAUDRATE = 115200
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)

# ====== 設定 ======
SAVE_CSV = True
CSV_PATH = "emg_stream_multi.csv"

FS = 200.0
RMS_WIN_MS = 20
RMS_WIN = max(3, int(FS * RMS_WIN_MS / 1000.0))
EMA_ALPHA = 0.2
K_SIGMA = 4.0
SLOPE_THR_FACTOR = 0.5
REFRACTORY_MS = 120
PEAK_WIN_MS = 80
MAX_MOTOR_WAIT = 0.5  # 500ms

# ====== チャンネル設定 ======
trigger_channels = [1, 2, 5, 6]  # ch2,ch3,ch6,ch7 -> 0始まり
peak_channels = [5, 6]  # ch6,ch7 -> 0始まり

# 各トリガーチャネルの重み (同じ長さにする)
trigger_weights = [0.5, 1.0, 0.8, 0.7]  # 例

# ====== グローバル ======
baseline_duration = 3.0
running = True
calibrated = False

baseline_trigger = [[] for _ in trigger_channels]
baseline_peak = [[] for _ in peak_channels]

mu_trigger = [0.0] * len(trigger_channels)
sigma_trigger = [1.0] * len(trigger_channels)
mu_peak = [0.0] * len(peak_channels)
sigma_peak = [1.0] * len(peak_channels)
Smax_peak = None

buf = np.zeros((RMS_WIN, 8), dtype=np.float32)
buf_ptr = 0
buf_filled = 0
sumsq = np.zeros(8, dtype=np.float32)

last_hit_time = 0.0
track_peak = False
track_until = 0.0
peak_S = 0.0
pending_hit_time = None

# CSV
csv_file = open(CSV_PATH, "w", newline="") if SAVE_CSV else None
csv_writer = csv.writer(csv_file) if csv_file else None
if csv_writer:
    csv_writer.writerow(
        [
            "trigger_time",
            "hit_time",
            "strength",
            "pc_delay_ms",
            "sw_delay_ms",
            "motor_delay_ms",
            "total_delay_ms",
        ]
    )

# HIT管理
last_trigger_time = None
last_hit_time = None
current_hit_row = None
motor_wait_start = None


# ==================== ユーティリティ ====================
def format_time(ts):
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


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


# ==================== EMG ハンドラ ====================
def on_emg(emg, movement):
    global calibrated, mu_trigger, sigma_trigger, mu_peak, sigma_peak, Smax_peak
    global last_hit_time, track_peak, track_until, peak_S, pending_hit_time
    global last_trigger_time, current_hit_row, motor_wait_start

    ts = time.time()
    x = np.asarray(emg, dtype=np.float32)

    # CSVに生データ保存（オプション）
    # if csv_writer:
    #     csv_writer.writerow([ts] + list(map(int, emg)))

    rms_vec = sliding_rms_update(x)
    if rms_vec is None:
        return

    rms_trigger = [rms_vec[ch] for ch in trigger_channels]
    rms_peak = [rms_vec[ch] for ch in peak_channels]

    # ===== キャリブレーション =====
    if not calibrated:
        for i in range(len(trigger_channels)):
            baseline_trigger[i].append(rms_trigger[i])
        for i in range(len(peak_channels)):
            baseline_peak[i].append(rms_peak[i])
        return

    now = ts

    # ===== ピーク追跡モード =====
    if track_peak:
        current_peak = max(rms_peak)
        if current_peak > peak_S:
            peak_S = current_peak
        if now >= track_until:
            if Smax_peak is None:
                Smax_peak = peak_S
            else:
                Smax_peak = max(Smax_peak * 0.995, Smax_peak)
                Smax_peak = max(Smax_peak, peak_S)

            # HIT強度
            lo = min([mu_peak[i] + K_SIGMA * sigma_peak[i] for i in range(len(peak_channels))])
            hi = max(lo + 1e-6, Smax_peak)
            strength = int(np.clip((peak_S - lo) / (hi - lo) * 100.0, 0, 100))

            hit_time = pending_hit_time if pending_hit_time is not None else now
            last_hit_time = hit_time

            pc_delay_ms = (hit_time - last_trigger_time) * 1000.0
            sw_delay_ms = pc_delay_ms

            # CSV用行作成
            current_hit_row = [
                format_time(last_trigger_time),
                format_time(hit_time),
                strength,
                f"{pc_delay_ms:.3f}",
                f"{sw_delay_ms:.3f}",
                "",
                "",  # motor_delay, total_delay
            ]
            motor_wait_start = now

            # Arduinoへ送信
            try:
                ser.write(f"HIT,{strength},{hit_time:.6f}\n".encode())
            except Exception as e:
                print("Arduino送信エラー:", e)

            track_peak = False
            pending_hit_time = None
        return

    # ===== 新しいHIT条件（トリガーチャネル） =====

    if (now - last_hit_time) * 1000.0 < REFRACTORY_MS:
        return

    # 各チャネルの標準化スコア（zスコアみたいなもの）
    trigger_scores = []
    for i in range(len(trigger_channels)):
        score = (rms_trigger[i] - mu_trigger[i]) / (sigma_trigger[i] + 1e-6)
        trigger_scores.append(score * trigger_weights[i])

    # 合算スコア
    trigger_sum = sum(trigger_scores)

    # 閾値（例: K_SIGMA を基準にする）
    threshold = K_SIGMA * sum(trigger_weights)

    if trigger_sum >= threshold:
        pending_hit_time = now
        last_trigger_time = pending_hit_time
        track_peak = True
        track_until = now + PEAK_WIN_MS / 1000.0
        peak_S = max(rms_peak)


# ==================== キャリブレーションと実行 ====================
def calibrate_and_run():
    global calibrated, mu_trigger, sigma_trigger, mu_peak, sigma_peak

    m = Myo(mode=emg_mode.RAW)
    m.connect()
    m.add_emg_handler(on_emg)

    print("=== キャリブレーション開始 ===")
    print(f"・腕をリラックスして静止してください（約{baseline_duration}秒）...")
    t0 = time.time()
    while (time.time() - t0) < baseline_duration:
        m.run()

    # キャリブ計算
    for i in range(len(trigger_channels)):
        arr = np.asarray(baseline_trigger[i], dtype=np.float32)
        mu_trigger[i] = float(np.mean(arr))
        sigma_trigger[i] = float(np.std(arr, ddof=1) + 1e-6)
    for i in range(len(peak_channels)):
        arr = np.asarray(baseline_peak[i], dtype=np.float32)
        mu_peak[i] = float(np.mean(arr))
        sigma_peak[i] = float(np.std(arr, ddof=1) + 1e-6)

    print("=== キャリブレーション完了 ===")
    for i, ch in enumerate(trigger_channels):
        print(f"Trigger ch{ch+1}: mu={mu_trigger[i]:.3f}, sigma={sigma_trigger[i]:.3f}")
    for i, ch in enumerate(peak_channels):
        print(f"Peak ch{ch+1}: mu={mu_peak[i]:.3f}, sigma={sigma_peak[i]:.3f}")

    calibrated = True
    print("=== 検出開始 === (Ctrl+Cで終了) ===")

    # Arduinoリスナー
    def arduino_listener():
        global current_hit_row, motor_wait_start
        while running:
            try:
                line = ser.readline().decode().strip()
                if line == "MOTOR" and current_hit_row is not None:
                    motor_time_pc = time.time()
                    motor_delay_ms = (motor_time_pc - last_hit_time) * 1000.0
                    total_delay_ms = (motor_time_pc - last_trigger_time) * 1000.0

                    current_hit_row[5] = f"{motor_delay_ms:.3f}"
                    current_hit_row[6] = f"{total_delay_ms:.3f}"

                    csv_writer.writerow(current_hit_row)
                    current_hit_row = None
                    motor_wait_start = None
            except Exception:
                pass

    listener_thread = threading.Thread(target=arduino_listener, daemon=True)
    listener_thread.start()

    # タイムアウトスレッド
    def timeout_checker():
        global current_hit_row, motor_wait_start
        while running:
            if current_hit_row is not None and motor_wait_start is not None:
                if time.time() - motor_wait_start > MAX_MOTOR_WAIT:
                    csv_writer.writerow(current_hit_row)
                    current_hit_row = None
                    motor_wait_start = None
            time.sleep(0.001)

    timeout_thread = threading.Thread(target=timeout_checker, daemon=True)
    timeout_thread.start()

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
