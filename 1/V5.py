import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
import glob  # 複数ファイル検索用

# --- 設定 ---
# ⚠️ データの保存先ディレクトリを設定
DATA_DIR = "./emg_data_raw/"
MODEL_FILE = "svm_drum_model_multi.joblib"
SCALER_FILE = "scaler_data_multi.joblib"

WINDOW_SIZE = 5  # RMS計算のためのサンプル数
M_SAMPLES = 10  # 差分特徴量の比較対象サンプル間隔

# EMGチャネルの選択 (Myoの8チャネルから必要なものを選ぶ)
# 例: 1と5 (CSVヘッダーの "CH1", "CH5" に対応)
CHANNELS = [1, 5]

# ラベル変換マップ
LABEL_MAP = {
    "rest": 0,
    "radial_dev": 1,  # 撓屈
    "ulnar_dev": 2,  # 尺屈
}


# --- 特徴量抽出関数 (修正版) ---
def extract_features(data, window_size, M, feature_cols):
    """
    data: データフレーム (CHx列とLabel列を含む)
    window_size: RMS計算窓幅
    M: 差分計算の間隔
    feature_cols: 計算に使用する信号列のリスト (例: ['CH1', 'CH5'])
    """
    N_samples = len(data)
    features = []

    # RMS計算と差分特徴量の生成
    for i in range(N_samples - window_size + 1):
        # ウィンドウの切り出し（全列）
        window_full = data.iloc[i : i + window_size]

        # ★修正: 計算には指定された feature_cols だけを使う（Label列を除外）
        window_signal = window_full[feature_cols]

        # 1. 絶対RMS (R_k)
        rms = np.sqrt(np.mean(window_signal**2, axis=0))
        feature_vector = rms.tolist()

        # 2. 差分RMS (ΔR_k) の計算
        if i >= M:
            past_window_full = data.iloc[i - M : i - M + window_size]
            past_window_signal = past_window_full[feature_cols]

            rms_past = np.sqrt(np.mean(past_window_signal**2, axis=0))
            delta_rms = rms - rms_past
            feature_vector.extend(delta_rms.tolist())
        else:
            # Mサンプル未満の場合はゼロでパディング
            feature_vector.extend([0.0] * len(feature_cols))

        features.append(feature_vector)

    # ラベルは特徴量抽出後の対応する位置のものを抽出
    # NOTE: 'Label'列が存在することを前提とする
    labels_series = data["Label"].iloc[window_size - 1 :].values
    labels_series = labels_series[M:]

    return np.array(features[M:]), labels_series


# --- 複数ファイルからのデータロード関数 ---
def load_data_from_directory(data_dir, channels, label_map):
    all_emg_data = []

    # ディレクトリ内のすべてのCSVファイルを検索
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    if not csv_files:
        print(f"エラー: {data_dir} 内にCSVファイルが見つかりません。")
        return None

    print(f"✅ {len(csv_files)}個のCSVファイルを読み込みます...")

    # 各ファイルを個別に読み込み、結合
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)

            # 必要なカラムのみを選択
            channel_cols = [f"CH{c}" for c in channels]
            required_cols = channel_cols + ["Label"]

            if not all(col in df.columns for col in required_cols):
                print(f"警告: {file_path} に必要なカラムが見つかりませんでした。スキップします。")
                continue

            df_selected = df[required_cols].copy()

            # 文字列ラベルを数値に変換し、不明なラベルを含む行を削除
            df_selected["Label"] = df_selected["Label"].map(label_map)
            df_selected.dropna(subset=["Label"], inplace=True)
            df_selected["Label"] = df_selected["Label"].astype(int)

            # EMGデータを結合リストに追加
            all_emg_data.append(df_selected)

        except Exception as e:
            print(f"ファイル {file_path} の処理中にエラーが発生しました: {e}")
            continue

    if not all_emg_data:
        print("エラー: 有効なデータが抽出されたCSVファイルがありませんでした。")
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
    # 計算対象の列名リストを作成 ('CH1', 'CH5'など)
    emg_cols = [f"CH{c}" for c in CHANNELS]

    # ★修正: renameを削除し、emg_colsを引数として渡す
    X, y = extract_features(combined_df, WINDOW_SIZE, M_SAMPLES, feature_cols=emg_cols)

    # 抽出後のデータサイズチェック
    if len(X) == 0:
        print("エラー: 特徴量抽出後のデータが空です。ウィンドウサイズやM_SAMPLESの設定を確認してください。")
        return

    print(f"特徴量抽出後のサンプル数: {len(X)}")

    # 3. データの正規化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. 学習/テストデータ分割
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    print(f"学習データ数: {len(X_train)}, テストデータ数: {len(X_test)}")

    # 5. SVMモデルの学習
    svm_model = SVC(kernel="linear", C=1.0)
    svm_model.fit(X_train, y_train)

    # 6. 評価
    y_pred = svm_model.predict(X_test)

    print("\n--- Classification Report ---")
    target_names = [name for name, val in sorted(LABEL_MAP.items(), key=lambda item: item[1])]
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # 7. モデルとスケーラーの保存
    joblib.dump(svm_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"\n--- Model and Scaler saved to {MODEL_FILE} and {SCALER_FILE} ---")


if __name__ == "__main__":
    train_model()
