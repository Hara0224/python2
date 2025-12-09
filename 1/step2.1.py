import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
import glob

# --- 設定 ---
# ⚠️ データの保存先ディレクトリを設定
DATA_DIR = "./emg_data_svm/"
MODEL_FILE = "svm_drum_model_multi.joblib"
SCALER_FILE = "scaler_data_multi.joblib"

WINDOW_SIZE = 5
M_SAMPLES = 10

# EMGチャネルの選択 (撓屈: 2, 3; 尺屈: 6, 7)
CHANNELS = [2, 3, 6, 7]

# ラベル変換マップ
LABEL_MAP = {
    "rest": 0,
    "radial_dev": 1,  # 撓屈
    "ulnar_dev": 2,  # 尺屈
}


# --- 特徴量抽出関数 (変更なし) ---
def extract_features(data, window_size, M, feature_cols):
    """
    data: データフレーム (CHx列とLabel列を含む)
    window_size: RMS計算窓幅
    M: 差分計算の間隔 (ΔR_kを計算するために必要な過去データのオフセット)
    feature_cols: 計算に使用する信号列のリスト (例: ['CH1', 'CH5'])
    """
    N_samples = len(data)
    features = []

    start_index = M + window_size - 1

    for i in range(start_index, N_samples):
        # 現在のウィンドウ
        current_window_signal = data.iloc[i - window_size + 1 : i + 1][feature_cols]

        # 過去のウィンドウ
        past_window_signal = data.iloc[i - window_size + 1 - M : i + 1 - M][
            feature_cols
        ]

        # 1. 絶対RMS (R_k)
        rms = np.sqrt(np.mean(current_window_signal**2, axis=0))
        feature_vector = rms.tolist()

        # 2. 差分RMS (ΔR_k)
        rms_past = np.sqrt(np.mean(past_window_signal**2, axis=0))
        delta_rms = rms - rms_past
        feature_vector.extend(delta_rms.tolist())

        features.append(feature_vector)

    # ラベルは、特徴量が計算された時点 (ウィンドウの末尾) のものを使用
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
                print(
                    f"警告: {file_path} に必要なカラムが見つかりませんでした。スキップします。"
                )
                continue

            df_selected = df[required_cols].copy()
            df_selected["Label"] = df_selected["Label"].map(label_map)
            df_selected.dropna(subset=["Label"], inplace=True)
            df_selected["Label"] = df_selected["Label"].astype(int)
            all_emg_data.append(df_selected)

        except Exception as e:
            print(f"ファイル {file_path} の処理中にエラーが発生しました: {e}")
            continue

    if not all_emg_data:
        print("エラー: 有効なデータが抽出されたCSVファイルがありませんでした。")
        return None

    return pd.concat(all_emg_data, ignore_index=True)


# --- メイン学習プロセス (修正箇所あり) ---
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
        print(
            "エラー: 特徴量抽出後のデータが空です。ウィンドウサイズやM_SAMPLESの設定を確認してください。"
        )
        return

    feature_names = [f"RMS_{col}" for col in emg_cols] + [
        f"DeltaRMS_{col}" for col in emg_cols
    ]

    print(f"特徴量抽出後のサンプル数: {len(X)}")
    print(f"生成された特徴量 ({len(feature_names)}次元): {', '.join(feature_names)}")

    # 2.5. クラス分布の確認
    print("\n--- クラス分布 ---")
    label_counts = pd.Series(y).value_counts().sort_index()
    label_map_rev = {v: k for k, v in LABEL_MAP.items()}
    for label_val, count in label_counts.items():
        print(f"  {label_map_rev.get(label_val, '不明')}: {count} サンプル")

    # 3. データの正規化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. 学習/テストデータ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n学習データ数: {len(X_train)}, テストデータ数: {len(X_test)}")

    # 5. SVMモデルの学習
    # 🌟 変更点: kernel="rbf", C=1.0 に設定
    print("\n--- SVMモデル学習開始 (Kernel: RBF, C: 1.0) ---")
    svm_model = SVC(kernel="rbf", C=1.0, random_state=42)
    svm_model.fit(X_train, y_train)

    # 6. 評価
    y_pred = svm_model.predict(X_test)

    print("\n--- Classification Report ---")
    target_names = [
        name for name, val in sorted(LABEL_MAP.items(), key=lambda item: item[1])
    ]
    print(
        classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        )
    )

    # 7. モデルとスケーラーの保存
    joblib.dump(svm_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"\n--- Model and Scaler saved to **{MODEL_FILE}** and **{SCALER_FILE}** ---")


if __name__ == "__main__":
    train_model()
