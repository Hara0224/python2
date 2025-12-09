import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# === 1. 設定パラメータ ===
SAVE_DIR = r"G:\マイドライブ\GooglePython\EMG\emg_data_svm"
MODEL_SAVE_PATH = "svm_onset_model_rms_tuned_v2.joblib"  # v2に変更
SCALER_SAVE_PATH = "scaler_onset_rms_tuned_v2.joblib"  # v2に変更

# 特徴量抽出パラメータ
CHANNELS = [2, 3, 6, 7]  # 使用するMyoチャンネル (CH1から数える)
FS = 200  # サンプリング周波数 (Hz)
WINDOW_MS = 100  # ウィンドウサイズ (ms)
STEP_MS = 25  # ステップサイズ (ms)
WINDOW_SAMPLES = int(FS * WINDOW_MS / 1000)  # 20 サンプル
STEP_SAMPLES = int(FS * STEP_MS / 1000)  # 5 サンプル
K_SIGMA = 3.5  # Onset検出のための標準偏差の倍率
ONSET_WEIGHT_MULTIPLIER = 10  # Onsetウィンドウに与える重み（高速応答性維持）
SUSTAINED_WEIGHT_MULTIPLIER = 2.0  # <<<<<< 変更点 1: 持続動作の重みを引き上げ >>>>>>

# SVMモデルパラメータ
SVM_PARAMS = {
    "kernel": "rbf",
    "C": 0.5,
    "gamma": "scale",
    # class_weight はカスタム辞書を後で適用するため、ここでは設定しない
}
# <<<<<< 変更点 2: restクラスの重みを抑制するカスタム辞書 >>>>>>
CUSTOM_CLASS_WEIGHT = {
    "radial_dev": 1.0,
    "ulnar_dev": 1.0,
    "rest": 1.0,  # restクラスの誤分類を防ぐためにわずかに重みを抑制
}


# === 2. 特徴量計算関数 (変更なし) ===


def calculate_rms(window_data):
    """ウィンドウデータからRMSを計算 (4チャンネル分)"""
    # RMS = sqrt( 1/N * sum(x^2) )
    return np.sqrt(np.mean(window_data**2, axis=0))


def process_emg_data(file_paths):
    """
    CSVファイルを読み込み、DCオフセット除去、RMS/Delta RMS特徴量計算、
    およびOnsetマーキングを行う。
    """
    all_data = []
    rest_emg_data = []  # DCオフセット・ノイズRMS基準計算用

    print("--- Phase 1: データロードと基準値計算 ---")

    for path in file_paths:
        df = pd.read_csv(path)
        all_data.append(df)

        if df["Label"].iloc[0] == "rest":
            # チャンネル名に合わせるため、CH{i}で抽出
            emg_cols = [f"CH{i}" for i in CHANNELS]
            rest_emg_data.append(df[emg_cols].values)

    # -----------------------------------------------------------
    # 2.1. DCオフセットとOnset閾値の計算
    # -----------------------------------------------------------

    if not rest_emg_data:
        raise ValueError("restラベルのデータが見つかりません。restデータが必要です。")

    rest_emg_combined = np.concatenate(rest_emg_data, axis=0)

    # チャンネルごとのDCオフセット (平均値) を計算
    dc_offset = np.mean(rest_emg_combined, axis=0)
    print(f"[INFO] DCオフセット: {dc_offset}")

    # RMSノイズ基準の計算 (Onset閾値 T_onset の計算用)
    rest_rms_values = []
    for i in range(0, len(rest_emg_combined) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        window = rest_emg_combined[i : i + WINDOW_SAMPLES]
        rms_window = calculate_rms(window - dc_offset)
        rest_rms_values.append(rms_window)

    rest_rms_flat = np.concatenate(rest_rms_values)
    mu_rest = np.mean(rest_rms_flat)
    sigma_rest = np.std(rest_rms_flat)
    T_onset = mu_rest + K_SIGMA * sigma_rest
    print(
        f"[INFO] Onset検出閾値 (T_onset, RMS): {T_onset:.4f} (μ:{mu_rest:.4f}, σ:{sigma_rest:.4f})"
    )

    # -----------------------------------------------------------
    # 2.2. 特徴量抽出とOnsetマーキング
    # -----------------------------------------------------------
    all_features_list = []

    print("\n--- Phase 2: 特徴量抽出とOnset重み付け ---")

    for df in all_data:
        label = df["Label"].iloc[0]
        emg_cols = [f"CH{i}" for i in CHANNELS]
        emg_data = df[emg_cols].values

        # 1. DCオフセット除去
        emg_data_offset_free = emg_data - dc_offset

        previous_rms = None  # Delta RMS計算用
        is_onset_detected = False

        # 2. スライディングウィンドウ処理
        for i in range(0, len(emg_data_offset_free) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            window = emg_data_offset_free[i : i + WINDOW_SAMPLES]

            # RMSを計算 (4次元)
            current_rms = calculate_rms(window)

            # Delta RMSを計算 (4次元)
            if previous_rms is None:
                delta_rms = np.zeros(len(CHANNELS))
            else:
                delta_rms = current_rms - previous_rms

            # 特徴ベクトル (8次元)
            current_features = np.concatenate((current_rms, delta_rms))

            # Onset/Transition判定と重み付けフラグ設定
            is_high_priority = 0  # 0: 通常の重み

            # <<<<<< 変更点 3: サンプル重みの計算ロジックを修正 >>>>>>
            if label == "rest":
                sample_weight_value = 1.0  # restは常に1.0
            else:
                # 動作データの場合
                if not is_onset_detected:
                    # 立ち上がり閾値 T_onset を超えたか
                    if np.any(current_rms > T_onset):
                        # Onsetウィンドウ: 高速応答のため最高重み
                        sample_weight_value = ONSET_WEIGHT_MULTIPLIER
                        is_onset_detected = True
                    else:
                        # Onset前: 静止中なので重み1.0
                        sample_weight_value = 1.0
                else:
                    # Onset検出後の持続動作: 途切れを防ぐため重みを上げる
                    sample_weight_value = SUSTAINED_WEIGHT_MULTIPLIER

            # 特徴量、ラベル、重みをリストに追加
            feature_row = list(current_features) + [label, sample_weight_value]
            all_features_list.append(feature_row)

            previous_rms = current_rms  # 次のDelta RMS計算のために保存

    return all_features_list


# === 3. メイン実行ブロック ===


def main():
    # -----------------------------------------------------------
    # A. データ処理と特徴量データセットの作成
    # -----------------------------------------------------------
    all_csv_files = [
        os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR) if f.endswith(".csv")
    ]

    if not all_csv_files:
        print(
            f"❌ '{SAVE_DIR}'内にCSVファイルが見つかりません。データ収集が完了しているか確認してください。"
        )
        return

    # 特徴量抽出とOnsetマーキングを実行
    all_features_list = process_emg_data(all_csv_files)

    # データフレームに変換
    feature_names = [f"RMS_CH{c}" for c in CHANNELS] + [
        f"DeltaRMS_CH{c}" for c in CHANNELS
    ]
    df_features = pd.DataFrame(
        all_features_list, columns=feature_names + ["Label", "SampleWeight"]
    )

    print(f"✅ 特徴量データセット生成完了: 総サンプル数 {len(df_features)}")

    # -----------------------------------------------------------
    # B. SVMモデルの学習と評価
    # -----------------------------------------------------------
    print("\n--- Phase 3: SVMモデルの学習と評価 ---")

    # データ分割
    X = df_features[feature_names].values
    y = df_features["Label"].values
    weights = df_features["SampleWeight"].values

    # テストデータと学習データを分割
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )

    print(f"学習データ数: {len(X_train)}, テストデータ数: {len(X_test)}")

    # 特徴量の標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # モデルの初期化と学習
    # <<<<<< 変更点 4: class_weightをカスタム辞書として渡す >>>>>>
    svm = SVC(**SVM_PARAMS, class_weight=CUSTOM_CLASS_WEIGHT, random_state=42)

    print("\n--- SVM Onsetモデル学習開始 (重み調整版) ---")
    # サンプル重みとクラス重みを組み合わせて学習
    svm.fit(X_train_scaled, y_train, sample_weight=weights_train)

    # 評価
    y_pred = svm.predict(X_test_scaled)

    print("\n--- SVM Onset Classification Report (テストデータ全体) ---")
    print(classification_report(y_test, y_pred, digits=4))

    # Onsetサンプルのみの評価
    onset_indices = np.where(weights_test == ONSET_WEIGHT_MULTIPLIER)[
        0
    ]  # Onsetサンプルは重み10
    if len(onset_indices) > 0:
        print(
            f"\n--- Onset/High Priority サンプル (重み {ONSET_WEIGHT_MULTIPLIER}.0) の評価 ---"
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
