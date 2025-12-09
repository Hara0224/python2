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
DATA_DIR = "./emg_data_svm/"
MODEL_FILE_RADIAL = "svm_radial_binary.joblib"  # Radialモデルのファイル名
SCALER_FILE_RADIAL = "scaler_radial_binary.joblib"  # Radialスケーラー
MODEL_FILE_ULNAR = "svm_ulnar_binary.joblib"  # Ulnarモデルのファイル名
SCALER_FILE_ULNAR = "scaler_ulnar_binary.joblib"  # Ulnarスケーラー

WINDOW_SIZE = 10
M_SAMPLES = 10

# 🌟 チャンネルの分離
ALL_CHANNELS = [2, 3, 6, 7]
RADIAL_CHANNELS = [2, 3]  # Radial Dev専用チャンネル
ULNAR_CHANNELS = [6, 7]  # Ulnar Dev専用チャンネル

# ラベル変換マップ
LABEL_MAP = {
    "rest": 0,
    "radial_dev": 1,
    "ulnar_dev": 2,
}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# ★ RBFカーネル設定
SVM_GAMMA = 0.1
SVM_C = 0.01


# --- 特徴量抽出関数 (WL追加版) ---
def extract_features(data, window_size, M, feature_cols):
    """RMS, DeltaRMS, WLの12次元特徴量を抽出する（チャネル指定可能）"""
    N_samples = len(data)
    features = []

    start_index = M + window_size - 1

    for i in range(start_index, N_samples):
        current_window_signal = data.iloc[i - window_size + 1 : i + 1][feature_cols]
        past_window_signal = data.iloc[i - window_size + 1 - M : i + 1 - M][
            feature_cols
        ]

        feature_vector = []

        # 1. 絶対RMS (R_k)
        rms = np.sqrt(np.mean(current_window_signal**2, axis=0))
        feature_vector.extend(rms.tolist())

        # 2. 差分RMS (ΔR_k)
        rms_past = np.sqrt(np.mean(past_window_signal**2, axis=0))
        delta_rms = rms - rms_past
        feature_vector.extend(delta_rms.tolist())

        # 3. WL (Waveform Length)
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
    # ... (データロード関数は省略し、以前のものと同じとします) ...
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


# --- 🌟 最終判定ロジック (推論関数) ---
def predict_parallel(score_rad, score_uln, rest_label=LABEL_MAP["rest"]):
    """
    並列分類器の決定スコアを統合し、最終ラベルを決定する。
    スコアはSVMのdecision_function (ハイパープレーンからの距離) を使用する。
    """
    y_pred = np.full(len(score_rad), rest_label)  # デフォルトはrest (0)

    # 1. radial_dev のスコアが高い場合
    # score_rad > score_uln (他の動作より撓屈の確信度が高い)
    # かつ score_rad > 0 (撓屈と予測された空間にある)
    # Ulnarのスコアも考慮し、拮抗している場合はrestにする

    # 🌟 判定閾値 (C=0.1の場合、決定境界は0だが、ノイズ対策で少し余裕を持たせる)
    # ここでは単純に 0 を使用します
    THRESHOLD = -0.5

    # 撓屈と予測する条件
    is_radial = (score_rad > THRESHOLD) & (score_rad > score_uln)

    # 尺屈と予測する条件
    is_ulnar = (score_uln > THRESHOLD) & (
        score_uln >= score_rad
    )  # >= はスコアが等しい場合に尺屈を優先 (任意)

    # ラベルを更新
    y_pred[is_radial] = LABEL_MAP["radial_dev"]
    y_pred[is_ulnar] = LABEL_MAP["ulnar_dev"]

    # 両方のスコアが閾値より低い場合、またはスコアが非常に近い場合は、デフォルトのrestのままとなる。
    # 例: score_rad=0.5, score_uln=0.6 の場合、ulnar_dev に分類される。
    # 例: score_rad=-0.1, score_uln=-0.2 の場合、rest に分類される。

    return y_pred


# --- メイン学習プロセス ---
def train_model():
    # 1. データロードと特徴量抽出
    combined_df = load_data_from_directory(DATA_DIR, ALL_CHANNELS, LABEL_MAP)
    if combined_df is None:
        return

    # 全チャンネルの特徴量 (RMS, DeltaRMS, WL) を抽出
    emg_cols = [f"CH{c}" for c in ALL_CHANNELS]
    X, y = extract_features(combined_df, WINDOW_SIZE, M_SAMPLES, feature_cols=emg_cols)

    feature_names = [f"{t}_{c}" for t in ["RMS", "DeltaRMS", "WL"] for c in emg_cols]

    print(f"特徴量抽出後のサンプル数: {len(X)}")
    print(f"生成された特徴量 ({len(feature_names)}次元): {', '.join(feature_names)}")

    # 2. 学習/テストデータ分割
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n学習データ数: {len(X_train_full)}, テストデータ数: {len(X_test_full)}")

    # 3. データの準備とモデルの学習

    # --- Radial Dev モデルの学習 ---
    print("\n--- Radial Dev (CH2, CH3) モデル学習開始 ---")

    # Radial Dev に必要な特徴量カラムを選択
    rad_features = [
        name for name in feature_names if any(f"CH{c}" in name for c in RADIAL_CHANNELS)
    ]

    # X_train_fullからRadial特徴量のみを抽出
    df_train_full = pd.DataFrame(X_train_full, columns=feature_names)
    X_rad_train_raw = df_train_full[rad_features].values

    # ラベルをバイナリに変換: radial_dev(1) vs other(0)
    y_rad_train = np.where(y_train == LABEL_MAP["radial_dev"], 1, 0)

    # スケーリング
    scaler_rad = StandardScaler()
    X_rad_train = scaler_rad.fit_transform(X_rad_train_raw)

    # モデル学習
    svm_rad = SVC(
        kernel="rbf", C=SVM_C, gamma=SVM_GAMMA, random_state=42, probability=False
    )  # decision_functionを使用するためprobability=False
    svm_rad.fit(X_rad_train, y_rad_train)

    joblib.dump(svm_rad, MODEL_FILE_RADIAL)
    joblib.dump(scaler_rad, SCALER_FILE_RADIAL)
    print(f"✅ Radial Dev モデルとスケーラーを保存: {MODEL_FILE_RADIAL}")

    # --- Ulnar Dev モデルの学習 ---
    print("\n--- Ulnar Dev (CH6, CH7) モデル学習開始 ---")

    # Ulnar Dev に必要な特徴量カラムを選択
    uln_features = [
        name for name in feature_names if any(f"CH{c}" in name for c in ULNAR_CHANNELS)
    ]

    # X_train_fullからUlnar特徴量のみを抽出
    X_uln_train_raw = df_train_full[uln_features].values

    # ラベルをバイナリに変換: ulnar_dev(1) vs other(0)
    y_uln_train = np.where(y_train == LABEL_MAP["ulnar_dev"], 1, 0)

    # スケーリング
    scaler_uln = StandardScaler()
    X_uln_train = scaler_uln.fit_transform(X_uln_train_raw)

    # モデル学習
    svm_uln = SVC(
        kernel="rbf", C=SVM_C, gamma=SVM_GAMMA, random_state=42, probability=False
    )
    svm_uln.fit(X_uln_train, y_uln_train)

    joblib.dump(svm_uln, MODEL_FILE_ULNAR)
    joblib.dump(scaler_uln, SCALER_FILE_ULNAR)
    print(f"✅ Ulnar Dev モデルとスケーラーを保存: {SCALER_FILE_ULNAR}")

    # 4. 評価 (テストデータ)
    print("\n--- 並列分類器による評価開始 ---")

    # テストデータから特徴量を抽出し、それぞれのスケーラーで正規化
    df_test_full = pd.DataFrame(X_test_full, columns=feature_names)

    X_rad_test_raw = df_test_full[rad_features].values
    X_rad_test = scaler_rad.transform(X_rad_test_raw)

    X_uln_test_raw = df_test_full[uln_features].values
    X_uln_test = scaler_uln.transform(X_uln_test_raw)

    # 各モデルから決定関数スコアを取得
    score_rad = svm_rad.decision_function(X_rad_test)
    score_uln = svm_uln.decision_function(X_uln_test)

    # 最終判定ロジックを適用
    y_pred_combined = predict_parallel(score_rad, score_uln)

    # 評価レポート
    print("\n--- Classification Report (並列分類器統合結果) ---")
    target_names = [REVERSE_LABEL_MAP[val] for val in sorted(REVERSE_LABEL_MAP.keys())]
    print(
        classification_report(
            y_test, y_pred_combined, target_names=target_names, zero_division=0
        )
    )

    print("\n--- 完了: 2つのモデルとスケーラーが保存されました ---")


if __name__ == "__main__":
    train_model()
