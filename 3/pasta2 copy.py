import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# === 1. 設定パラメータ ===
SAVE_DIR = "./emg_data_raw/"
MODEL_SAVE_PATH = "svm_onset_model_rms_tuned_v6.joblib"  # V6
SCALER_SAVE_PATH = "scaler_onset_rms_tuned_v6.joblib"  # V6
FEATURE_DATA_PATH = "svm_input_features_rms_v5.csv"  # 特徴量データセットはv5のものを再利用

# 特徴量抽出パラメータ
CHANNELS = [2, 3, 6, 7]
FS = 200
WINDOW_MS = 100
STEP_MS = 25
WINDOW_SAMPLES = int(FS * WINDOW_MS / 1000)
STEP_SAMPLES = int(FS * STEP_MS / 1000)
K_SIGMA = 3.5
ONSET_WEIGHT_MULTIPLIER = 10
# V5の設定を維持
SUSTAINED_RADIAL_WEIGHT = 4.0
SUSTAINED_ULNAR_WEIGHT = 3.0

# SVMモデルパラメータ
SVM_PARAMS = {
    "kernel": "rbf",
    "C": 0.5,
    "gamma": "scale",
}
# <<<<<< 変更点 1: restクラスの重みを2.0に引き上げる >>>>>>
CUSTOM_CLASS_WEIGHT = {"radial_dev": 1.0, "ulnar_dev": 1.0, "rest": 2.0}
FEATURE_NAMES = [f"RMS_CH{c}" for c in CHANNELS] + [f"DeltaRMS_CH{c}" for c in CHANNELS]


# === 2. 特徴量計算関数 (省略 - 特徴量ファイルは既にv5で作成済みのため) ===

# 特徴量抽出関数は省略し、データロードと学習に移行します。


def main():
    # -----------------------------------------------------------
    # A. データ処理と特徴量データセットのロード
    # -----------------------------------------------------------
    print("--- Phase 1 & 2: 特徴量データセットのロード (V5結果を利用) ---")
    try:
        df_features = pd.read_csv(FEATURE_DATA_PATH)
        print(f"✅ 特徴量データセットロード完了: {FEATURE_DATA_PATH} (総サンプル数 {len(df_features)})")
    except FileNotFoundError:
        print(f"❌ 特徴量データセットファイルが見つかりません: {FEATURE_DATA_PATH}")
        print("特徴量データセットを再作成してください。")
        return

    # -----------------------------------------------------------
    # B. SVMモデルの学習と評価
    # -----------------------------------------------------------
    print("\n--- Phase 3: SVMモデルの学習と評価 (V6: Rest安定化版) ---")

    X = df_features[FEATURE_NAMES].values
    y = df_features["Label"].values
    weights = df_features["SampleWeight"].values

    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42, stratify=y)

    print(f"学習データ数: {len(X_train)}, テストデータ数: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # モデルの初期化と学習
    # <<<<<< 変更点 2: CUSTOM_CLASS_WEIGHTを適用 >>>>>>
    svm = SVC(**SVM_PARAMS, class_weight=CUSTOM_CLASS_WEIGHT, random_state=42)

    print("\n--- SVM Onsetモデル学習開始 (V6: Rest安定化版) ---")
    # サンプル重みとクラス重みを組み合わせて学習
    svm.fit(X_train_scaled, y_train, sample_weight=weights_train)

    # 評価
    y_pred = svm.predict(X_test_scaled)

    print("\n--- SVM Onset Classification Report (テストデータ全体) ---")
    print(classification_report(y_test, y_pred, digits=4))

    # Onsetサンプルのみの評価
    onset_indices = np.where(weights_test == ONSET_WEIGHT_MULTIPLIER)[0]
    if len(onset_indices) > 0:
        print(f"\n--- Onset/High Priority サンプル (重み {ONSET_WEIGHT_MULTIPLIER}.0) の評価 ---")
        print(classification_report(y_test[onset_indices], y_pred[onset_indices], digits=4))
    else:
        print("\n[INFO] テストデータ中にOnset/High Priorityサンプルがありませんでした。")

    # -----------------------------------------------------------
    # C. モデルとスケーラーの保存
    # -----------------------------------------------------------
    joblib.dump(svm, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"\n✅ モデルとスケーラーを保存しました: {MODEL_SAVE_PATH}, {SCALER_SAVE_PATH}")


if __name__ == "__main__":
    main()
