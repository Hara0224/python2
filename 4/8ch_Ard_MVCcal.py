import time
import csv
import sys
import threading
import numpy as np

# ====== Myo ======
from pyomyo import Myo, emg_mode

# ====== （任意）Arduino送信用 ======
# import serial
# ser = serial.Serial('COM5', 115200)  # 環境に合わせて

# ====== 設定 ======
SAVE_CSV = True
CSV_PATH = "emg_stream_log1.csv"

FS = 200.0  # Myoは約200Hz
DT = 1.0 / FS
RMS_WIN_MS = 20  # 低遅延のため短め
RMS_WIN = max(3, int(FS * RMS_WIN_MS / 1000.0))
EMA_ALPHA = 0.2  # 因果平滑（遅延ほぼ0）
K_SIGMA = 4.0  # 動的しきい値倍率
SLOPE_THR_FACTOR = 0.5  # 傾き=0.5σ0 目安
REFRACTORY_MS = 120
PEAK_WIN_MS = 80

# チャネル重み（最初は均等: 8chとも1.0）
CHANNEL_WEIGHTS = np.array([0.8, 0.8, 1.2, 1.0, 0.5, 1.2, 1.0, 0.5], dtype=np.float32)

# ====== グローバル状態 ======
lock = threading.Lock()
running = True
calibrated = False

# キャリブ用
baseline_samples = []
baseline_duration = 3.0  # 秒（静止）
mvc_samples = []  # 強めにスナップ 2秒目安（任意）

mu0 = 0.0
sigma0 = 1.0
slope_thr = 0.1
Smax_cal = None  # 強度正規化の上限（MVC相当）

# 移動RMS用リングバッファ
buf = np.zeros((RMS_WIN, 8), dtype=np.float32)
buf_ptr = 0
buf_filled = 0
sumsq = np.zeros(8, dtype=np.float32)

# EMA用
ema_S = 0.0
ema_initialized = False

# 検出状態
last_hit_time = 0.0
track_peak = False
track_until = 0.0
peak_S = 0.0
pending_hit_time = None

# CSV
csv_file = open(CSV_PATH, "w", newline="") if SAVE_CSV else None
csv_writer = csv.writer(csv_file) if csv_file else None
if csv_writer:
    csv_writer.writerow(["Timestamp"] + [f"ch{i+1}" for i in range(8)])


def sliding_rms_update(x):  # x: shape (8,)
    global buf, buf_ptr, buf_filled, sumsq
    old = buf[buf_ptr]
    sumsq += x * x - old * old
    buf[buf_ptr] = x
    buf_ptr = (buf_ptr + 1) % RMS_WIN
    buf_filled = min(buf_filled + 1, RMS_WIN)
    if buf_filled < RMS_WIN:
        return None  # まだ窓が埋まってない
    return np.sqrt(sumsq / RMS_WIN)


def combine_channels(rms_vec):
    # 重み付き和 → スカラー活性度S
    return float(np.dot(rms_vec, CHANNEL_WEIGHTS) / CHANNEL_WEIGHTS.sum())


def update_ema(value):
    global ema_S, ema_initialized
    if not ema_initialized:
        ema_S = value
        ema_initialized = True
    else:
        ema_S = EMA_ALPHA * value + (1.0 - EMA_ALPHA) * ema_S
    return ema_S


def on_emg(emg, movement):
    # emg: list[int] 長さ8（-128〜127程度）
    global running, baseline_samples, mvc_samples, calibrated
    global mu0, sigma0, slope_thr, Smax_cal
    global ema_S, ema_initialized
    global last_hit_time, track_peak, track_until, peak_S, pending_hit_time

    ts = time.time()
    x = np.asarray(emg, dtype=np.float32)

    # ===== CSV保存 =====
    if csv_writer:
        csv_writer.writerow([ts] + list(map(int, emg)))

    # ===== 移動RMS（因果・1サンプル更新）=====
    rms_vec = sliding_rms_update(x)
    if rms_vec is None:
        return

    # ===== チャネル融合 =====
    S_raw = combine_channels(rms_vec)

    # ===== EMA平滑（遅延ほぼ0）=====
    S = update_ema(S_raw)

    # ===== キャリブ収集中はデータを貯めるのみ =====
    if not calibrated:
        baseline_samples.append(S)
        return

    # ===== 動的しきい値＆傾き判定 =====
    T = mu0 + K_SIGMA * sigma0

    # 立ち上がり傾き（単純差分）
    # EMAなので差分でOK
    # 過去値は ema_S に蓄積されているので、ここでは近似として (S - ema_S_prev) は使えないため、
    # 簡易に「直近の生値との差分」を別途保存しても良い。ここでは S_raw を使って近似。
    # 高速＆簡易化のため、S - T を使った即時傾き近似を入れる：
    slope_est = S - T  # T付近での相対値（>0なら上向き）

    now = ts
    if track_peak:
        # オンセット後のピーク探索期間
        if S > peak_S:
            peak_S = S
        if now >= track_until:
            # 強度決定
            if Smax_cal is None:
                # MVCが未設定なら、動的に学習（上限を広げすぎないように）
                Smax_cal = peak_S
            else:
                Smax_cal = max(
                    Smax_cal * 0.995, Smax_cal
                )  # 緩やかに減衰して過適応を避ける
                Smax_cal = max(Smax_cal, peak_S)

            # 0-100% 正規化
            lo = T
            hi = max(T + 1e-6, Smax_cal)  # 0除算回避
            strength = int(np.clip((peak_S - lo) / (hi - lo) * 100.0, 0, 100))

            # 出力（ここがArduino送信ポイント）
            hit_time = pending_hit_time if pending_hit_time is not None else now
            print(
                f"[HIT] t={hit_time:.3f} strength={strength:3d}  peak_S={peak_S:.3f}  T={T:.3f}"
            )

            # 例：Arduinoに送る（有効化する場合は上のserialを開く）
            # ser.write(f"HIT,{strength}\n".encode())

            track_peak = False
            last_hit_time = now
            pending_hit_time = None
        return

    # 不応期
    if (now - last_hit_time) * 1000.0 < REFRACTORY_MS:
        return

    # しきい値クロス＋傾き
    if (S >= T) and (slope_est >= SLOPE_THR_FACTOR * sigma0):
        # オンセット確定
        pending_hit_time = now
        track_peak = True
        track_until = now + PEAK_WIN_MS / 1000.0
        peak_S = S


def calibrate_and_run():
    global calibrated, baseline_samples, mu0, sigma0, slope_thr

    m = Myo(mode=emg_mode.RAW)
    m.connect()
    m.add_emg_handler(on_emg)

    print("=== キャリブレーション開始 ===")
    print("・腕をリラックスして静止してください（約3秒）...")
    t0 = time.time()
    while (time.time() - t0) < baseline_duration:
        m.run()

    # 基線統計
    arr = np.asarray(baseline_samples, dtype=np.float32)
    if arr.size < 10:
        print("キャリブ用データが不足")
        sys.exit(1)
    mu0 = float(np.mean(arr))
    sigma0 = float(np.std(arr, ddof=1) + 1e-6)
    print(f"基線: mu0={mu0:.3f}, sigma0={sigma0:.3f}")

    # 傾きしきい値（相対）
    print(
        f"傾きしきい値は {SLOPE_THR_FACTOR} * sigma0 = {SLOPE_THR_FACTOR * sigma0:.3f}"
    )

    calibrated = True
    print("=== 検出開始：スナップするとHITが表示されます（Ctrl+Cで終了） ===")

    try:
        while True:
            m.run()
    except KeyboardInterrupt:
        pass
    finally:
        if csv_file:
            csv_file.close()
        m.disconnect()
        print("終了")


if __name__ == "__main__":
    calibrate_and_run()
