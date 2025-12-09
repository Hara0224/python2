import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
import glob

# --- ⚙️ 設定 ---
DATA_DIR = r"G:\マイドライブ\GooglePython\EMG\emg_data_svm"
MODEL_FILE = "svm_onset_model_custom_w.joblib"  # モデルファイル名をカスタム重み版に変更
SCALER_FILE = "scaler_onset_svm_custom_w.joblib"

WINDOW_SIZE = 10
M_SAMPLES = 10
CHANNELS = [2, 6]
FEATURE_COLS = [f"CH{c}" for c in CHANNELS]

# ★ Onset定義パラメータ
ONSET_POSITIVE_WINDOWS = 5
ONSET_SD_MULTIPLIER = 1.5  # ノイズ閾値を緩和

# SVMパラメータ
SVM_C = 0.1  # 🌟 Cを0.1に設定してノイズ耐性を向上
SVM_GAMMA = "scale"

# ★ 🌟 修正点: カスタムクラス重み (Precision改善が目標)
CUSTOM_CLASS_WEIGHTS = {
    0: 1.0,  # rest (基準)
    1: 25.0,  # radial_dev (調整。balancedの約131から大きく引き下げ)
    2: 25.0,  # ulnar_dev (調整)
}

LABEL_MAP = {
    "rest": 0,
    "radial_dev": 1,
    "ulnar_dev": 2,
}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# --- 📊 特徴量抽出関数 (安定化済み) ---
def extract_features(data, window_size, M, feature_cols):
    """RMSとDeltaRMSの8次元特徴量を抽出する"""
    N_samples = len(data)
    features = []

    # 🚨 型変換とNaN処理
    data[feature_cols] = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    data.dropna(subset=feature_cols, inplace=True)
    N_samples = len(data)

    start_index = M + window_size - 1

    if N_samples <= start_index:
        return pd.DataFrame(), np.array([])

    for i in range(start_index, N_samples):
        window_full = data.iloc[i - window_size + 1 : i + 1]
        window_signal = window_full[feature_cols]

        rms = np.sqrt(np.mean(window_signal**2, axis=0))

        if i >= M + window_size:
            past_window_full = data.iloc[i - M - window_size + 1 : i - M + 1]
            past_window_signal = past_window_full[feature_cols]

            rms_past = np.sqrt(np.mean(past_window_signal**2, axis=0))
            delta_rms = rms - rms_past
        else:
            delta_rms = np.zeros_like(rms)

        feature_vector = rms.tolist() + delta_rms.tolist()
        features.append(feature_vector)

    labels_series = data["Label"].iloc[start_index:].values
    return (
        pd.DataFrame(
            features,
            columns=[f"{t}_{c}" for t in ["RMS", "DeltaRMS"] for c in feature_cols],
        ),
        labels_series,
    )


# --- 💾 複数ファイルからのデータロード関数 (以前と同一) ---
def load_data_from_directory(data_dir, channels, label_map):
    all_emg_data = []
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        return None

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            channel_cols = [f"CH{c}" for c in channels]
            required_cols = channel_cols + ["Label"]
            if not all(col in df.columns for col in required_cols):
                continue

            df_selected = df[required_cols].copy()
            df_selected["Label"] = df_selected["Label"].map(label_map)
            df_selected.dropna(subset=["Label"], inplace=True)
            df_selected["Label"] = df_selected["Label"].astype(int)
            all_emg_data.append(df_selected)
        except Exception:
            continue

    if not all_emg_data:
        return None
    return pd.concat(all_emg_data, ignore_index=True)


# --- 🌟 データセット再ラベリング関数 (Onset特化) ---
def relabel_for_onset(df, window_size, M, sd_multiplier, feature_cols):

    X_raw, y_raw = extract_features(df, window_size, M, feature_cols)

    if len(X_raw) == 0:
        return pd.DataFrame(), np.array([])

    X_raw["Label"] = y_raw

    rest_data = X_raw[X_raw["Label"] == LABEL_MAP["rest"]]
    if len(rest_data) == 0:
        return X_raw.drop(columns=["Label"]), y_raw

    rest_rms_cols = [f"RMS_CH{c}" for c in CHANNELS]

    M_rms = rest_data[rest_rms_cols].mean().mean()
    SD_rms = rest_data[rest_rms_cols].values.std()

    T_noise = M_rms + sd_multiplier * SD_rms
    print(f"ノイズレベル閾値 (T_noise): {T_noise:.4f} (平均:{M_rms:.4f}, SD:{SD_rms:.4f})")

    new_labels = np.full(len(X_raw), LABEL_MAP["rest"], dtype=int)

    current_label = LABEL_MAP["rest"]
    onset_started = False

    for i in range(len(X_raw)):
        row = X_raw.iloc[i]
        label = row["Label"]
        current_rms_mean = row[rest_rms_cols].mean()

        if label != current_label:
            current_label = label
            onset_started = False
            onset_window_count = 0

        if current_label != LABEL_MAP["rest"]:

            if not onset_started and current_rms_mean > T_noise:
                onset_started = True
                onset_window_count = 1
                new_labels[i] = current_label

            elif onset_started and onset_window_count < ONSET_POSITIVE_WINDOWS:
                onset_window_count += 1
                new_labels[i] = current_label

            elif onset_started and onset_window_count >= ONSET_POSITIVE_WINDOWS:
                new_labels[i] = LABEL_MAP["rest"]

    X_new = X_raw.drop(columns=["Label"])
    y_new = new_labels

    print(f"新しいPositiveサンプル数: {np.sum(y_new != 0)} (全体の {np.sum(y_new != 0) / len(y_new) * 100:.2f}%)")
    return X_new, y_new


# --- 🧪 メイン学習プロセス (Onset SVM) ---
def relabel_and_train_onset_svm():
    # 1. データロード
    combined_df = load_data_from_directory(DATA_DIR, CHANNELS, LABEL_MAP)
    if combined_df is None:
        return

    # 2. Onset特化ラベリングと特徴量抽出の実行
    X_onset, y_onset = relabel_for_onset(combined_df, WINDOW_SIZE, M_SAMPLES, ONSET_SD_MULTIPLIER, FEATURE_COLS)

    if len(X_onset) == 0:
        return

    # 3. データの分割
    X_train, X_test, y_train, y_test = train_test_split(X_onset.values, y_onset, test_size=0.2, random_state=42, stratify=y_onset)

    # 4. データの正規化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. SVMモデルの学習
    print(f"\n--- SVM Onsetモデル学習開始 (C: {SVM_C}, Gamma: {SVM_GAMMA}, Weighted: Custom {CUSTOM_CLASS_WEIGHTS[1]}) ---")

    # 🌟 Custom Class Weightを適用
    svm_model = SVC(
        kernel="rbf",
        C=SVM_C,
        gamma=SVM_GAMMA,
        class_weight=CUSTOM_CLASS_WEIGHTS,
        random_state=42,
    )
    svm_model.fit(X_train_scaled, y_train)

    # 6. 評価
    y_pred = svm_model.predict(X_test_scaled)

    print("\n--- SVM Onset Classification Report ---")
    target_names = [name for name, val in sorted(LABEL_MAP.items(), key=lambda item: item[1])]
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # 7. モデルの保存
    joblib.dump(svm_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"\n--- Model and Scaler saved to {MODEL_FILE} and {SCALER_FILE} ---")


if __name__ == "__main__":
    relabel_and_train_onset_svm()
