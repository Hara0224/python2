import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os
import glob

# --- 設定 ---
DATA_DIR = r"G:\マイドライブ\GooglePython\EMG\emg_data_svm"
MODEL_FILE = "rf_drum_model_final.joblib"
SCALER_FILE = "scaler_rf_final.joblib"

# 即応性を維持するため WINDOW_SIZE=10 を採用
WINDOW_SIZE = 10
M_SAMPLES = 10

# EMGチャネルの選択
CHANNELS = [2, 3, 6, 7]

# ラベル変換マップ
LABEL_MAP = {
    "rest": 0,
    "radial_dev": 1,  # 撓屈
    "ulnar_dev": 2,  # 尺屈
}

# ★ Random Forest ハイパーパラメータ
RF_ESTIMATORS = 100

# ★ クラス重み (Radial Devの検出力を強化)
CLASS_WEIGHTS = {0: 0.8, 1: 1.5, 2: 1.0}


# --- 特徴量抽出関数 (RMS, DeltaRMS, WL の 3種類) ---
def extract_features(data, window_size, M, feature_cols):
    """
    RMS, DeltaRMS, WL の合計12次元特徴量を抽出
    即応性を高めるため、計算コストの低いWLを採用
    """
    N_samples = len(data)
    features = []

    start_index = M + window_size - 1

    for i in range(start_index, N_samples):
        current_window_signal = data.iloc[i - window_size + 1 : i + 1][feature_cols]
        past_window_signal = data.iloc[i - window_size + 1 - M : i + 1 - M][
            feature_cols
        ]

        feature_vector = []

        # 1. 絶対RMS (R_k) - 信号のパワー
        rms = np.sqrt(np.mean(current_window_signal**2, axis=0))
        feature_vector.extend(rms.tolist())

        # 2. 差分RMS (ΔR_k) - 動きの立ち上がり検出
        rms_past = np.sqrt(np.mean(past_window_signal**2, axis=0))
        delta_rms = rms - rms_past
        feature_vector.extend(delta_rms.tolist())

        # 3. WL (Waveform Length) - 信号の複雑さと瞬発力
        # WL = Σ|x[i+1] - x[i]|
        wl_list = []
        for col in feature_cols:
            signal = current_window_signal[col].values
            wl = np.sum(np.abs(np.diff(signal)))
            wl_list.append(wl)

        feature_vector.extend(wl_list)
        features.append(feature_vector)

    labels_series = data["Label"].iloc[start_index:].values

    return np.array(features), labels_series


# --- 複数ファイルからのデータロード関数 (変更なし) ---
def load_data_from_directory(data_dir, channels, label_map):
    all_emg_data = []
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    if not csv_files:
        print(f"エラー: {data_dir} 内にCSVファイルが見つかりません。")
        return None

    print(f"✅ {len(csv_files)}個のCSVファイルを読み込みます...")

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            channel_cols = [f"CH{c}" for c in channels]
            required_cols = channel_cols + ["Label"]

            if not all(col in df.columns for col in required_cols):
                # print(f"警告: {file_path} ...スキップ")
                continue

            df_selected = df[required_cols].copy()
            df_selected["Label"] = df_selected["Label"].map(label_map)
            df_selected.dropna(subset=["Label"], inplace=True)
            df_selected["Label"] = df_selected["Label"].astype(int)
            all_emg_data.append(df_selected)

        except Exception as e:
            print(f"エラー: {file_path} - {e}")
            continue

    if not all_emg_data:
        print("エラー: 有効なデータがありません。")
        return None

    return pd.concat(all_emg_data, ignore_index=True)


# --- メイン学習プロセス ---
def train_model():
    # 1. データロード
    combined_df = load_data_from_directory(DATA_DIR, CHANNELS, LABEL_MAP)
    if combined_df is None:
        return

    print(f"総データサンプル数: {len(combined_df)}")

    # 2. 特徴量抽出
    emg_cols = [f"CH{c}" for c in CHANNELS]
    X, y = extract_features(combined_df, WINDOW_SIZE, M_SAMPLES, feature_cols=emg_cols)
    if len(X) == 0:
        return

    # 特徴量名のリスト (4ch * 3種類 = 12次元)
    feature_names = (
        [f"RMS_{col}" for col in emg_cols]
        + [f"DeltaRMS_{col}" for col in emg_cols]
        + [f"WL_{col}" for col in emg_cols]
    )

    print(f"特徴量抽出後のサンプル数: {len(X)}")
    print(f"生成された特徴量 ({len(feature_names)}次元): {', '.join(feature_names)}")

    # 2.5. クラス分布
    print("\n--- クラス分布 ---")
    label_counts = pd.Series(y).value_counts().sort_index()
    label_map_rev = {v: k for k, v in LABEL_MAP.items()}
    for label_val, count in label_counts.items():
        print(f"  {label_map_rev.get(label_val, '不明')}: {count} サンプル")

    X_processed = X

    # 4. 学習/テストデータ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n学習データ数: {len(X_train)}, テストデータ数: {len(X_test)}")

    # 5. Random Forest モデルの学習
    print("\n--- Random Forest モデル学習開始 (Weighted + WL特徴量) ---")

    rf_model = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        random_state=42,
        n_jobs=-1,
        class_weight=CLASS_WEIGHTS,  # 重み付け適用
    )
    rf_model.fit(X_train, y_train)

    # 6. 評価
    y_pred = rf_model.predict(X_test)

    print("\n--- Classification Report ---")
    target_names = [
        name for name, val in sorted(LABEL_MAP.items(), key=lambda item: item[1])
    ]
    print(
        classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        )
    )

    # 7. モデルの保存
    joblib.dump(rf_model, MODEL_FILE)
    print(f"\n--- Model saved to **{MODEL_FILE}** ---")


if __name__ == "__main__":
    train_model()
