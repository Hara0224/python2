import numpy as np
import time
import joblib
import multiprocessing
import queue
import sys
import os
from pyomyo import Myo, emg_mode
import collections

# === 1. V12モデルの設定とロード ===
MODEL_PATH = "svm_hybrid_v12_delta.joblib"
SCALER_PATH = "scaler_hybrid_v12_delta.joblib"

# 特徴量抽出パラメータ
CHANNELS = [2, 3, 6, 7]
FS = 200  # サンプリング周波数 (Hz)
STEP_MS = 5  # 制御周期 (ms)
WINDOW_MS = 50  # 特徴量抽出ウィンドウサイズ (ms)

WINDOW_SAMPLES = int(FS * WINDOW_MS / 1000)  # 10 サンプル
STEP_SAMPLES = int(FS * STEP_MS / 1000)  # 5 サンプル <--- 追加

# V12学習時に計算された値
DC_OFFSET = np.array([-0.57694795, -0.56773359, -0.63077918, -0.47289579])
RADIAL_INDICES = [0, 1]  # CH2, CH3
ULNAR_INDICES = [2, 3]  # CH6, CH7

# モデルとスケーラーのロード (メインプロセスでロード)
try:
    SVM_MODEL = joblib.load(MODEL_PATH)
    SCALER = joblib.load(SCALER_PATH)
except FileNotFoundError:
    print(f"❌ モデルファイルが見つかりません: {MODEL_PATH} または {SCALER_PATH}")
    sys.exit()

# === 2. グローバル変数とキュー ===
EMG_QUEUE = multiprocessing.Queue()
EMG_BUFFER = collections.deque(maxlen=WINDOW_SAMPLES)
PREVIOUS_RMS = None
IS_RUNNING = True


# === 3. 補助関数 ===
def initialize_buffer():
    for _ in range(WINDOW_SAMPLES):
        EMG_BUFFER.append(np.zeros(8))


def calculate_rms(window_data):
    return np.sqrt(np.mean(window_data**2, axis=0))


def calculate_features(emg_window):
    global PREVIOUS_RMS

    selected_indices = [c - 1 for c in CHANNELS]
    selected_emg = emg_window[:, selected_indices]

    offset_free_emg = selected_emg - DC_OFFSET
    current_rms = calculate_rms(offset_free_emg)

    if PREVIOUS_RMS is None:
        delta_rms = np.zeros(len(CHANNELS))
    else:
        delta_rms = current_rms - PREVIOUS_RMS

    features = np.concatenate((current_rms, delta_rms))
    PREVIOUS_RMS = current_rms

    X_scaled = SCALER.transform([features])
    return X_scaled, current_rms


# === 4. Myoプロセス (EMGデータ取得) ===


def myo_worker(emg_q):
    m = Myo(mode=emg_mode.RAW)
    try:
        m.connect()
    except Exception as e:
        print(f"Worker Error: Myo接続失敗: {e}")
        return

    def add_to_queue(emg, movement):
        emg_q.put(emg)

    m.set_leds([0, 128, 0], [0, 0, 0])
    m.vibrate(1)
    m.add_emg_handler(add_to_queue)

    while True:
        try:
            # Myoイベントを処理し、add_to_queueにデータを送る
            m.run()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Worker Exception: {e}")
            break
    m.disconnect()
    print("Myo Worker Stopped")


# === 5. 制御ロジック関数 ===


def control_output(prediction, current_rms):
    if prediction == "rest":
        print("-> ⚪ [rest] 静止/保持")
        return

    radial_rms_avg = np.mean(current_rms[RADIAL_INDICES])
    ulnar_rms_avg = np.mean(current_rms[ULNAR_INDICES])

    magnitude = max(0.0, radial_rms_avg + ulnar_rms_avg) * 0.1

    if radial_rms_avg > ulnar_rms_avg:
        direction = "radial_dev"
        print(f"-> 🔴 [{direction}] 撓屈動作実行 (強度: {magnitude:.2f})")
    else:
        direction = "ulnar_dev"
        print(f"-> 🔵 [{direction}] 尺屈動作実行 (強度: {magnitude:.2f})")


# === 6. リアルタイム処理ループ (メインプロセス) ===


def real_time_control_loop():
    global IS_RUNNING

    CONTROL_PERIOD = STEP_MS / 1000.0  # 0.025秒

    while IS_RUNNING:
        start_time = time.time()

        # 1. EMGデータ更新 (キューから全て取り込み、EMG_BUFFERを更新)
        while not EMG_QUEUE.empty():
            emg_data = EMG_QUEUE.get_nowait()
            EMG_BUFFER.append(np.array(emg_data))

        # 2. 制御処理を実行
        # Window size (50ms)を満たしているかチェック
        if len(EMG_BUFFER) >= WINDOW_SAMPLES:

            emg_window = np.array(list(EMG_BUFFER))

            try:
                X_scaled, current_rms = calculate_features(emg_window)
            except Exception as e:
                print(f"❌ 特徴量計算エラー: {e}")
                continue

            prediction = SVM_MODEL.predict(X_scaled)[0]
            control_output(prediction, current_rms)

        # 3. 制御周期に合わせて待機 (25ms周期を確保)
        elapsed_time = time.time() - start_time
        sleep_time = CONTROL_PERIOD - elapsed_time
        if sleep_time > 0:
            # 制御周期の残りを待機し、次のループへ
            time.sleep(sleep_time)


# === 7. メイン実行 ===

if __name__ == "__main__":

    initialize_buffer()

    # 1. Myoプロセスを開始
    p = multiprocessing.Process(target=myo_worker, args=(EMG_QUEUE,))
    p.start()

    print("🟢 V17リアルタイム制御を開始。Myo接続待機中...")

    # 2. キューにデータが来るまで待機
    while EMG_QUEUE.empty():
        time.sleep(0.1)

    print("✅ Myoからのデータ受信開始。制御ループ実行中。")

    try:
        real_time_control_loop()

    except KeyboardInterrupt:
        print("\nプログラムを終了します。")
    finally:
        IS_RUNNING = False
        if p.is_alive():
            p.terminate()  # Myoプロセスを強制終了
        p.join()
        print("システムシャットダウン完了。")
