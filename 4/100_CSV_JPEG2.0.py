import os
import time
import csv
import threading
import signal
import cv2
import numpy as np
from scipy.signal import butter, lfilter, iirnotch
from datetime import datetime
from pyomyo import Myo, emg_mode

# === 設定 ===
labels = ["1"]
repeats_per_label = 1
record_duration = 2.0
fps = 30
interval_between = 1.0

emg_save_dir = "./emg_data_raw4/"
img_save_dir = "./img_data_raw4/"
os.makedirs(emg_save_dir, exist_ok=True)
os.makedirs(img_save_dir, exist_ok=True)

# === Myo 初期化 ===
m = Myo(mode=emg_mode.RAW)
m.connect()
m.set_leds([128, 0, 0], [0, 0, 0])
m.vibrate(1)

# === グローバル変数 ===
raw_data = []
raw_data_lock = threading.Lock()
first_emg_time = None
first_emg_lock = threading.Lock()
stop_flag = False


# === Ctrl+C ハンドラ ===
def signal_handler(sig, frame):
    global stop_flag
    stop_flag = True
    print("\n🛑 Ctrl+C を検出しました。安全に停止します…")


signal.signal(signal.SIGINT, signal_handler)

# === フィルター設計 ===
fs = 200
nyq = 0.5 * fs
b_band, a_band = butter(4, [20 / nyq, 90 / nyq], btype="band")
b_notch, a_notch = iirnotch(50 / nyq, Q=30)
b_lp, a_lp = butter(4, 4 / nyq, btype="low")


# === リアルタイム EMG 処理関数 ===
def process_emg_envelope(emg):
    x = lfilter(b_band, a_band, emg)
    x = lfilter(b_notch, a_notch, x)
    x = np.abs(x)
    x = lfilter(b_lp, a_lp, x)
    return x


# === EMG ハンドラ（置き換え部分） ===
def collect_emg(emg, movement):
    global first_emg_time
    ts = time.time()
    if isinstance(emg, (list, tuple)) and len(emg) == 8:
        emg_np = np.array(emg, dtype=np.float32)
        processed = process_emg_envelope(emg_np)
        with raw_data_lock:
            raw_data.append([ts] + processed.tolist())
        with first_emg_lock:
            if first_emg_time is None:
                first_emg_time = ts


m.add_emg_handler(collect_emg)


# === 画像キャプチャ関数 ===
def capture_images(label, session_id, cap):
    global first_emg_time, stop_flag
    img_dir = os.path.join(img_save_dir, f"{label}_{session_id}")
    os.makedirs(img_dir, exist_ok=True)
    print("🕒 EMG受信開始を待っています...")
    while not stop_flag:
        with first_emg_lock:
            if first_emg_time is not None:
                start_time = first_emg_time
                break
        time.sleep(0.001)
    print(f"🕒 EMG受信開始検出: {start_time:.6f}秒。画像撮影開始…")
    frame_count = int(record_duration * fps)
    interval = 1.0 / fps
    for i in range(frame_count):
        if stop_flag:
            break
        target_time = start_time + i * interval
        sleep_time = target_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        ret, frame = cap.read()
        if not ret:
            print("⚠️ フレーム取得失敗")
            continue
        elapsed = time.time() - start_time
        elapsed_str = f"{elapsed:.3f}"
        cv2.putText(
            frame, elapsed_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )
        filename = os.path.join(img_dir, f"{elapsed_str.replace('.', '')}.jpg")
        cv2.imwrite(filename, frame)


# === カメラ初期設定 ===
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ カメラが開けません")
    exit()
print("🎥 カメラをウォームアップ中…")
for _ in range(10):
    cap.read()
    time.sleep(0.05)
print("✅ カメラ準備完了")

# === メイン記録ループ ===
print(
    f"🟢 記録準備完了。{record_duration}秒 × 各ラベル × {repeats_per_label}回 記録します。"
)
counter = 1
total = len(labels) * repeats_per_label

try:
    for _ in range(repeats_per_label):
        for label in labels:
            if stop_flag:
                break
            print(f"\n🔴 [{counter}/{total}] ラベル：{label} を記録中…")
            m.vibrate(2)
            time.sleep(2)
            with raw_data_lock:
                raw_data.clear()
            with first_emg_lock:
                first_emg_time = None
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_thread = threading.Thread(
                target=capture_images, args=(label, session_id, cap), daemon=True
            )
            img_thread.start()
            record_end = time.time() + record_duration
            while time.time() < record_end and not stop_flag:
                m.run()
            img_thread.join()
            filename = os.path.join(emg_save_dir, f"emgenv_{label}_{session_id}.csv")
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp"] + [f"CH{i+1}" for i in range(8)])
                with raw_data_lock:
                    writer.writerows(raw_data)
            print(f"✅ EMG保存完了: {filename} (サンプル: {len(raw_data)})")
            counter += 1
            time.sleep(interval_between)
except KeyboardInterrupt:
    print("\n🛑 手動で停止されました")
finally:
    m.vibrate(3)
    m.disconnect()
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ 終了処理完了。お疲れさまでした！")
