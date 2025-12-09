import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import glob

# === 1. 設定パラメータ (V11: T_ONSET 閾値引き下げ) ===
SAVE_DIR = r"G:\マイドライブ\GooglePython\EMG\emg_data_svm"
MODEL_SAVE_PATH = "svm_hybrid_v11.joblib"  # V11
SCALER_SAVE_PATH = "scaler_hybrid_v11.joblib"  # V11
FEATURE_DATA_PATH = "svm_input_features_hybrid_v9.csv"  # V9データを再利用

# 特徴量抽出パラメータ
CHANNELS = [2, 3, 6, 7]
FS = 200
WINDOW_MS = 50
STEP_MS = 25
WINDOW_SAMPLES = int(FS * WINDOW_MS / 1000)
STEP_SAMPLES = int(FS * STEP_MS / 1000)
# <<<< 変更点 1: 閾値乗数を 3.5 から 3.0 に引き下げ >>>>
K_SIGMA = 3.0
ONSET_WEIGHT_MULTIPLIER = 10.0
SUSTAINED_MOVEMENT_WEIGHT = 4.0

# SVMモデルパラメータ (V10設定を維持)
SVM_PARAMS = {"kernel": "rbf", "C": 0.5, "gamma": "scale"}
CUSTOM_CLASS_WEIGHT = {"Movement": 1.0, "rest": 4.0}
FEATURE_NAMES = [f"RMS_CH{c}" for c in CHANNELS] + [f"DeltaRMS_CH{c}" for c in CHANNELS]


# === 2. 特徴量抽出関数 (K_SIGMA 変更を適用) ===
def calculate_rms(window_data):
    return np.sqrt(np.mean(window_data**2, axis=0))


def process_emg_data(file_paths):
    all_data = []
    rest_emg_data = []

    print("--- Phase 1: データロードと基準値計算 (Window Size 50ms, K=3.0) ---")

    for path in file_paths:
        df = pd.read_csv(path)
        all_data.append(df)

        if df["Label"].iloc[0] == "rest":
            emg_cols = [f"CH{i}" for i in CHANNELS]
            rest_emg_data.append(df[emg_cols].values)

    if not rest_emg_data:
        raise ValueError("restラベルのデータが見つかりません。")

    rest_emg_combined = np.concatenate(rest_emg_data, axis=0)

    dc_offset = np.mean(rest_emg_combined, axis=0)
    print(f"[INFO] DCオフセット: {dc_offset}")

    # RMSノイズ基準の計算 (50msウィンドウを使用)
    rest_rms_values = []
    for i in range(0, len(rest_emg_combined) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        window = rest_emg_combined[i : i + WINDOW_SAMPLES]
        rms_window = calculate_rms(window - dc_offset)
        rest_rms_values.append(rms_window)

    rest_rms_flat = np.concatenate(rest_rms_values)
    mu_rest = np.mean(rest_rms_flat)
    sigma_rest = np.std(rest_rms_flat)
    # <<<< 変更点 2: K_SIGMA (3.0) を使用してT_onsetを再計算 >>>>
    T_onset = mu_rest + K_SIGMA * sigma_rest
    print(
        f"[INFO] Onset検出閾値 (T_onset, RMS): {T_onset:.4f} (μ:{mu_rest:.4f}, σ:{sigma_rest:.4f}, K={K_SIGMA})"
    )

    all_features_list = []
    print("\n--- Phase 2: 特徴量抽出と2クラスラベル付け ---")

    for df in all_data:
        label = df["Label"].iloc[0]
        emg_cols = [f"CH{i}" for i in CHANNELS]
        emg_data = df[emg_cols].values

        emg_data_offset_free = emg_data - dc_offset
        previous_rms = None
        is_onset_detected = False

        for i in range(0, len(emg_data_offset_free) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            window = emg_data_offset_free[i : i + WINDOW_SAMPLES]

            current_rms = calculate_rms(window)

            if previous_rms is None:
                delta_rms = np.zeros(len(CHANNELS))
            else:
                delta_rms = current_rms - previous_rms

            current_features = np.concatenate((current_rms, delta_rms))

            if label == "rest":
                new_label = "rest"
                sample_weight_value = 1.0
            else:
                new_label = "Movement"

                if not is_onset_detected:
                    if np.any(current_rms > T_onset):
                        sample_weight_value = ONSET_WEIGHT_MULTIPLIER
                        is_onset_detected = True
                    else:
                        sample_weight_value = 1.0
                else:
                    sample_weight_value = SUSTAINED_MOVEMENT_WEIGHT

            feature_row = list(current_features) + [new_label, sample_weight_value]
            all_features_list.append(feature_row)

            previous_rms = current_rms

    df_features = pd.DataFrame(
        all_features_list, columns=FEATURE_NAMES + ["Label", "SampleWeight"]
    )
    return df_features


# === 3. メイン実行ブロック (学習と評価) ===


def main():
    # CSVファイル検索
    all_csv_files = glob.glob(os.path.join(SAVE_DIR, "*.csv"))
    if not all_csv_files:
        print(f"❌ '{SAVE_DIR}'内にCSVファイルが見つかりません。")
        return

    df_features = process_emg_data(all_csv_files)

    # 特徴量データセットをCSVファイルとして保存 (V11)
    df_features.to_csv(FEATURE_DATA_PATH, index=False)
    print(
        f"✅ 特徴量データセット生成完了: {FEATURE_DATA_PATH} (総サンプル数 {len(df_features)})"
    )

    # -----------------------------------------------------------
    # B. SVMモデルの学習と評価 (V11: K=3.0)
    # -----------------------------------------------------------
    print("\n--- Phase 3: SVMモデルの学習と評価 (V11: K=3.0) ---")

    X = df_features[FEATURE_NAMES].values
    y = df_features["Label"].values
    weights = df_features["SampleWeight"].values

    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )

    print(f"学習データ数: {len(X_train)}, テストデータ数: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm = SVC(**SVM_PARAMS, class_weight=CUSTOM_CLASS_WEIGHT, random_state=42)

    print("\n--- SVM Onsetモデル学習開始 (V11: K=3.0) ---")
    svm.fit(X_train_scaled, y_train, sample_weight=weights_train)

    y_pred = svm.predict(X_test_scaled)

    print("\n--- SVM Onset Classification Report (テストデータ全体) ---")
    print(classification_report(y_test, y_pred, digits=4))

    # Onsetサンプルのみの評価
    onset_indices = np.where(weights_test == ONSET_WEIGHT_MULTIPLIER)[0]
    if len(onset_indices) > 0:
        print(
            f"\n--- Onset/High Priority サンプル (重み {ONSET_WEIGHT_MULTIPLIER}) の評価 ---"
        )
        print(
            classification_report(
                y_test[onset_indices], y_pred[onset_indices], digits=4
            )
        )
    else:
        print(
            "\n[INFO] テストデータ中にOnset/High Priorityサンプルがありませんでした。"
        )

    # -----------------------------------------------------------
    # C. モデルとスケーラーの保存
    # -----------------------------------------------------------
    joblib.dump(svm, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(
        f"\n✅ モデルとスケーラーを保存しました: {MODEL_SAVE_PATH}, {SCALER_SAVE_PATH}"
    )


if __name__ == "__main__":
    main()
