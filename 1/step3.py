import numpy as np
import joblib
import time
from collections import deque
from pyomyo import Myo, emg_mode

# --- 設定 ---
MODEL_FILE = "svm_drum_model_multi.joblib"
SCALER_FILE = "scaler_data_multi.joblib"
# ユーザー指定: RAWモードで接続
EMG_MODE = emg_mode.RAW

# train_svm.pyと一致させる
WINDOW_SIZE = 5
M_SAMPLES = 10
CHANNELS = [1, 5]  # 使用するチャネル番号 (1-8)

# --- グローバル変数 ---
emg_buffer = deque()
svm_model = None
scaler = None
labels = {0: "静止 (Rest)", 1: "撓屈 (Radial Dev)", 2: "尺屈 (Ulnar Dev)"}
last_prediction = -1
last_classification_time = 0.0

# 特徴量に必要なデータポイント数
REQUIRED_DATA_POINTS = len(CHANNELS) * (M_SAMPLES + WINDOW_SIZE)


# --- 特徴量抽出と分類（on_emgとして定義） ---
def on_emg(emg, moving_avg):
    global last_prediction, last_classification_time

    # モデルがロードされていない場合は処理をスキップ
    if svm_model is None or scaler is None:
        return

    # 1. 選択チャネルを抽出
    current_emg = np.array([emg[i - 1] for i in CHANNELS])

    # 2. バッファに新しいデータを追加
    for val in current_emg:
        emg_buffer.append(val)

    if len(emg_buffer) >= REQUIRED_DATA_POINTS:

        # 3. 必要な履歴データが揃ったら、最も古いデータポイントを削除
        if len(emg_buffer) > REQUIRED_DATA_POINTS:
            emg_buffer.popleft()

        # 4. 特徴量計算のためのデータ準備
        hist_array = np.array(emg_buffer).reshape(M_SAMPLES + WINDOW_SIZE, len(CHANNELS))

        # 最新のRMS計算 (R_k)
        latest_window = hist_array[-WINDOW_SIZE:]
        rms_latest = np.sqrt(np.mean(latest_window**2, axis=0))

        # 過去のRMS計算 (R_{k-M})
        past_window = hist_array[-(WINDOW_SIZE + M_SAMPLES) : -M_SAMPLES]
        rms_past = np.sqrt(np.mean(past_window**2, axis=0))

        # 特徴ベクトル生成: [R_k, R_k, ΔR_k, ΔR_k]
        delta_rms = rms_latest - rms_past
        feature_vector = np.concatenate([rms_latest, delta_rms]).reshape(1, -1)

        # 5. スケーリングと分類
        X_scaled = scaler.transform(feature_vector)
        prediction = svm_model.predict(X_scaled)[0]

        # 6. 予測結果が変化した場合のみ出力
        current_time = time.time()
        if prediction != last_prediction:
            latency = (current_time - last_classification_time) * 1000  # 遅延をミリ秒で計算
            print(f"[{current_time:.3f}s] -> PREDICTED: {labels.get(prediction)} | Latency since last: {latency:.2f}ms")
            last_prediction = prediction
            last_classification_time = current_time


# --- メイン実行 ---
def run_classifier():
    global svm_model, scaler, last_classification_time

    # モデルとスケーラーのロード
    try:
        svm_model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
    except FileNotFoundError:
        print("エラー: 学習済みモデルとスケーラーファイルが見つかりません。")
        print("`train_svm.py` を実行してファイルを生成してください。")
        return

    # ユーザー指定の接続コード
    m = Myo(emg_mode=EMG_MODE)
    m.add_emg_handler(on_emg)
    m.connect()
    print("[INFO] Myo接続完了")

    m.run_in_background(True)
    last_classification_time = time.time()  # 初期時刻を設定

    try:
        while m.is_connected():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        m.stop()


if __name__ == "__main__":
    run_classifier()
