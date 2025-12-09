import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import glob

# === 1. 設定パラメータ (V10: Final Stability Adjustment) ===
SAVE_DIR = r"G:\マイドライブ\GooglePython\EMG\emg_data_svm"
MODEL_SAVE_PATH = "svm_hybrid_v10.joblib"  # V10
SCALER_SAVE_PATH = "scaler_hybrid_v10.joblib"  # V10
FEATURE_DATA_PATH = "svm_input_features_hybrid_v9.csv"  # V9データを再利用

# 特徴量抽出パラメータ (V9設定を維持: 50ms Window)
CHANNELS = [2, 3, 6, 7]
FS = 200
WINDOW_MS = 50
STEP_MS = 25
WINDOW_SAMPLES = int(FS * WINDOW_MS / 1000)
STEP_SAMPLES = int(FS * STEP_MS / 1000)
K_SIGMA = 3.5
ONSET_WEIGHT_MULTIPLIER = 10.0
SUSTAINED_MOVEMENT_WEIGHT = 4.0

# SVMモデルパラメータ
SVM_PARAMS = {"kernel": "rbf", "C": 0.5, "gamma": "scale"}
# <<<< 変更点 1: restクラス重みを 4.0 に強化 >>>>
CUSTOM_CLASS_WEIGHT = {"Movement": 1.0, "rest": 4.0}
FEATURE_NAMES = [f"RMS_CH{c}" for c in CHANNELS] + [f"DeltaRMS_CH{c}" for c in CHANNELS]


# === 2. 特徴量抽出関数 (V9ロジックを簡略化して再利用) ===


# V9で生成された特徴量ファイルが存在することを前提とします。
def load_v9_data():
    try:
        df_features = pd.read_csv(FEATURE_DATA_PATH)
        return df_features
    except FileNotFoundError:
        # V9データがない場合は、V9の処理ロジック全体をここに含める必要があります。
        # (ここではファイルが存在すると仮定して処理を進めます)
        raise FileNotFoundError(f"V9の特徴量データファイル {FEATURE_DATA_PATH} が見つかりません。")


# === 3. メイン実行ブロック (学習と評価) ===


def main():
    df_features = load_v9_data()

    print("\n--- Phase 3: SVMモデルの学習と評価 (V10: Rest安定化最終版) ---")

    X = df_features[FEATURE_NAMES].values
    y = df_features["Label"].values
    weights = df_features["SampleWeight"].values

    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42, stratify=y)

    print(f"学習データ数: {len(X_train)}, テストデータ数: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm = SVC(**SVM_PARAMS, class_weight=CUSTOM_CLASS_WEIGHT, random_state=42)

    print("\n--- SVM Onsetモデル学習開始 (V10: Rest安定化) ---")
    svm.fit(X_train_scaled, y_train, sample_weight=weights_train)

    y_pred = svm.predict(X_test_scaled)

    print("\n--- SVM Onset Classification Report (テストデータ全体) ---")
    print(classification_report(y_test, y_pred, digits=4))

    # Onsetサンプルのみの評価
    onset_indices = np.where(weights_test == ONSET_WEIGHT_MULTIPLIER)[0]
    if len(onset_indices) > 0:
        print(f"\n--- Onset/High Priority サンプル (重み {ONSET_WEIGHT_MULTIPLIER}) の評価 ---")
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
