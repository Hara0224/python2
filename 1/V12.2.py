import numpy as np
import time
import joblib
from pyomyo import Myo, emg_mode
import collections
import sys
import os
from collections import Counter

# === 1. 設定パラメータと定数 (省略) ===
MODEL_PATH = "svm_hybrid_v12_delta.joblib"
SCALER_PATH = "scaler_hybrid_v12_delta.joblib"
CHANNELS = [2, 3, 6, 7]
FS = 200
STEP_MS = 5
WINDOW_MS = 50
WINDOW_SAMPLES = int(FS * WINDOW_MS / 1000)
STEP_SAMPLES = int(FS * STEP_MS / 1000)
DC_OFFSET = np.array([-0.57694795, -0.56773359, -0.63077918, -0.47289579])
RADIAL_INDICES = [0, 1]
ULNAR_INDICES = [2, 3]

# === 2. グローバル変数と初期化 ===
EMG_BUFFER = collections.deque(maxlen=WINDOW_SAMPLES)
PREVIOUS_RMS = None
IS_RUNNING = True
LAST_PREDICTION = "rest"


def initialize_buffer():
    for _ in range(WINDOW_SAMPLES):
        EMG_BUFFER.append(np.zeros(8))


# === 3. RMS計算関数群 ===
def calculate_rms(window_data):
    return np.sqrt(np.mean(window_data**2, axis=0))


def calculate_features(emg_window):
    # V21修正点: global宣言を追加
    global PREVIOUS_RMS

    selected_indices = [c - 1 for c in CHANNELS]
    selected_emg = emg_window[:, selected_indices]

    offset_free_emg = selected_emg - DC_OFFSET
    current_rms = calculate_rms(offset_free_emg)

    # Delta RMSの計算
    if PREVIOUS_RMS is None:
        delta_rms = np.zeros(len(CHANNELS))
    else:
        # PREVIOUS_RMSが存在する場合、差分を計算
        delta_rms = current_rms - PREVIOUS_RMS

    features = np.concatenate((current_rms, delta_rms))

    # PREVIOUS_RMSを更新
    PREVIOUS_RMS = current_rms

    # 標準化と次元調整 (SVM入力用: (1, 8))
    X_scaled = SCALER.transform([features])
    return X_scaled, current_rms


# === 4. データハンドラ関数 (MyoからのEMG受信) ===
def collect_emg(emg, movement):
    global EMG_BUFFER, IS_RUNNING
    if IS_RUNNING:
        EMG_BUFFER.append(np.array(emg))


# === 5. 制御出力関数 (ハイブリッド制御ロジック - V21出力変更ロジック) ===
def control_output(prediction, current_rms):
    global LAST_PREDICTION

    # 1. 状態がrestである場合の出力決定
    if prediction == "rest":
        if LAST_PREDICTION != "rest":
            print("-> ⚪ [rest] 静止/保持")
            LAST_PREDICTION = "rest"
        return  # 状態がrestで変化がない場合は何もしない

    # 2. 状態がMovementである場合の方向決定 (ステージ 2)
    radial_rms_avg = np.mean(current_rms[RADIAL_INDICES])
    ulnar_rms_avg = np.mean(current_rms[ULNAR_INDICES])

    # 比例制御用の強さ
    magnitude = max(0.0, radial_rms_avg + ulnar_rms_avg) * 0.1

    if radial_rms_avg > ulnar_rms_avg:
        direction = "radial_dev"
    else:
        direction = "ulnar_dev"

    # 3. 変化チェックと出力
    if direction != LAST_PREDICTION:
        print(
            f"-> {'🔴' if direction == 'radial_dev' else '🔵'} [{direction}] 動作実行 (強度: {magnitude:.2f})"
        )
        LAST_PREDICTION = direction  # 状態を更新
        # --- 義手アクチュエータへの出力コードをここに追加 ---


# === 6. リアルタイム処理ループ ===


def real_time_loop():
    global IS_RUNNING

    if len(EMG_BUFFER) < WINDOW_SAMPLES:
        return

    emg_window = np.array(list(EMG_BUFFER))

    # 1. 特徴量抽出と標準化
    try:
        # V21修正点: calculate_featuresがPREVIOUS_RMSを更新する
        X_scaled, current_rms = calculate_features(emg_window)
    except Exception as e:
        print(f"❌ 特徴量計算エラー: {e}")
        return

    # 2. ステージ 1: SVM推論 (Movement or rest)
    prediction = SVM_MODEL.predict(X_scaled)[0]

    # 3. V20/V21: 予測が直前の状態と異なる場合にのみ出力
    control_output(prediction, current_rms)


# === 7. Myo接続と実行 (メインブロック) ===

if __name__ == "__main__":

    # モデルとスケーラーのロード
    try:
        SVM_MODEL = joblib.load(MODEL_PATH)
        SCALER = joblib.load(SCALER_PATH)
        print("✅ モデルとスケーラーのロード完了。")
    except FileNotFoundError:
        print(f"❌ モデルファイルが見つかりません: {MODEL_PATH} または {SCALER_PATH}")
        sys.exit()

    initialize_buffer()

    m = Myo(mode=emg_mode.RAW)
    m.add_emg_handler(collect_emg)

    try:
        print("📡 Myoデバイスに接続を試行中...")
        m.connect()
    except Exception as e:
        print(f"❌ Myo接続エラー。デバイスが見つからないか、接続に失敗しました: {e}")
        sys.exit()

    print("\n🟢 Myo接続完了。V21リアルタイム制御を開始します。（出力変化検出ON）")
    m.set_leds([0, 128, 0], [0, 0, 0])

    CONTROL_PERIOD = STEP_MS / 1000.0  # 0.025秒 (25ms)

    try:
        while IS_RUNNING:
            start_time = time.time()

            m.run()

            real_time_loop()

            # 制御周期に合わせて待機
            elapsed_time = time.time() - start_time
            sleep_time = CONTROL_PERIOD - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nプログラムを終了します。")
    finally:
        IS_RUNNING = False
        m.disconnect()
