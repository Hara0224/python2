import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import japanize_matplotlib
import numpy as np

# ===== 設定 =====
input_folder = r"C:\Users\hrsyn\Desktop\PyT\emg_data_multiprocess"
csv_files = glob.glob(
    os.path.join(input_folder, "emgraw_*.csv")
)  # 'emgraw_' で始まるファイルに限定
target_ch = "CH3"
window_size = 10  # RMS/EMA計算に使用するウィンドウサイズ (例: 50サンプル)

# ===== グラフ描画（既存の描画ロジック） =====


def plot_all_files_default(csv_list, target_ch="CH3"):
    """既存の全ファイル描画ロジックを関数化"""

    # ylimをファイル名ごとに指定（必要なものだけ書く）
    ylim_settings = {
        # 注意: 既存のファイル名は 'emgraw_label_timestamp.csv' の形式です
        "2emg_rms.csv": (0, 100),
        "3emg_ema.csv": (0, 100),
        # 指定がないファイルはデフォルトで (-150, 150) を使う
    }

    plt.figure(figsize=(10, len(csv_list) * 3))

    for idx, file_path in enumerate(csv_list, start=1):
        df = pd.read_csv(file_path)
        # タイムスタンプをゼロ基準に変換
        time_s = df["Timestamp"] - df["Timestamp"].iloc[0]
        fname = os.path.basename(file_path)

        ax = plt.subplot(len(csv_list), 1, idx)
        ax.plot(time_s, df[target_ch], label=fname)
        ax.set_ylabel(f"{target_ch} Amplitude")

        # ylimをファイルごとに切り替え
        if fname in ylim_settings:
            ax.set_ylim(ylim_settings[fname])
        else:
            ax.set_ylim(-150, 150)  # デフォルト

        ax.set_title(f"{target_ch} - {fname}")
        ax.grid(True)
        if idx == len(csv_list):
            ax.set_xlabel("Time (s)")

    plt.subplots_adjust(hspace=0.5, top=0.95, bottom=0.05)
    plt.suptitle(f"Channel Extracted: {target_ch} (RAW Data)", fontsize=14)
    # plt.show() # この関数内では表示をスキップし、最後にまとめて表示


# --- グラフ描画（既存ロジックここまで） ---

# ===== 💡 追加システム：オンセット解析機能 =====


def plot_emg_delta_onset_analysis(
    input_folder, sample_label, num_trials, target_ch="CH3", window_size=50
):
    """
    指定されたラベルと試行回数に一致するファイルを検索し、
    EMG活動のRMSおよびオンセット検出の分析結果をプロットする。
    """
    print(f"\n--- 💡 EMGオンセット解析開始: {sample_label}, 試行数: {num_trials} ---")

    # 1. ファイルの検索とフィルタリング
    # ファイル名パターン: emgraw_[label]_[timestamp].csv
    search_pattern = f"emgraw_{sample_label}_*.csv"
    matching_files = glob.glob(os.path.join(input_folder, search_pattern))

    # ファイルをタイムスタンプ順（ファイル名順）にソート
    matching_files.sort()

    # 必要な試行回数分だけファイルを選択
    if len(matching_files) < num_trials:
        print(
            f"⚠️ 警告: {sample_label} のファイルが {len(matching_files)} 個しか見つかりませんでした。{num_trials} 個必要です。"
        )
        selected_files = matching_files
    else:
        selected_files = matching_files[:num_trials]

    if not selected_files:
        print(f"❌ ファイルが見つかりませんでした: {search_pattern}")
        return

    # 2. プロットの準備
    # 試行ごとに1つのサブプロットを作成 (各試行でRMSとRMSデルタを表示)
    plt.figure(figsize=(12, num_trials * 3))

    # 3. データ処理とプロット
    for idx, file_path in enumerate(selected_files, start=1):
        df = pd.read_csv(file_path)
        data = df[target_ch].values
        time_s = df["Timestamp"] - df["Timestamp"].iloc[0]

        # (A) RMS (二乗平均平方根) 計算
        # 絶対値を取り、二乗して移動平均をとり、平方根をとる
        rms = (
            pd.Series(data)
            .abs()
            .pow(2)
            .rolling(window=window_size, center=False)
            .mean()
            .pow(0.5)
        )

        # (B) RMS 変化率 (デルタ) 計算
        # デルタは活動開始を鋭敏に検出するのに役立つ
        rms_delta = rms.diff().rolling(window=window_size // 2, center=False).mean()

        # グラフ描画
        ax1 = plt.subplot(num_trials, 1, idx)

        # RMSプロット
        ax1.plot(time_s, rms, label="RMS (Activity)", color="C0")
        ax1.set_ylabel("RMS Amplitude", color="C0")
        ax1.tick_params(axis="y", labelcolor="C0")
        ax1.set_ylim(0, max(150, rms.max() * 1.2))

        # RMSデルタを重ねてプロット (活動開始の検出に利用)
        ax2 = ax1.twinx()
        ax2.plot(time_s, rms_delta, label="RMS Delta", color="C1", linestyle="--")
        ax2.set_ylabel("RMS Delta", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        ax2.axhline(y=0, color="gray", linestyle=":")

        # オンセット閾値の簡易的な設定 (例: RMSデルタが10を超えた場合)
        # 閾値はデータセットに応じて調整が必要です
        onset_threshold = 10
        ax2.axhline(
            y=onset_threshold,
            color="r",
            linestyle="-.",
            label=f"Onset Threshold ({onset_threshold})",
        )

        # 検出点のマーキング（プロットの装飾）
        onset_indices = rms_delta[rms_delta > onset_threshold].dropna().index
        if not onset_indices.empty:
            first_onset_index = onset_indices[0]
            onset_time = time_s.iloc[first_onset_index]
            ax1.axvline(
                x=onset_time,
                color="r",
                linestyle="-",
                linewidth=2,
                label=f"Onset at {onset_time:.2f}s",
            )

        ax1.set_title(f"{sample_label} - Trial {idx}: {os.path.basename(file_path)}")
        ax1.grid(True)
        if idx == num_trials:
            ax1.set_xlabel("Time (s)")

    plt.suptitle(f"EMG Onset Analysis: {sample_label} on {target_ch}", fontsize=16)
    plt.subplots_adjust(hspace=0.6, top=0.95, bottom=0.05)
    plt.show()  # ここでプロットを表示


# ===== メイン実行部分 =====

# 1. 既存の全ファイル描画ロジックを実行したい場合 (非推奨。ファイル数が多すぎるため)
# plot_all_files_default(csv_files, target_ch=target_ch)

# 2. 💡 要求されたオンセット解析システムを追加して実行
plot_emg_delta_onset_analysis(
    input_folder=input_folder,
    sample_label="radial_dev",
    num_trials=1,
    target_ch=target_ch,
)
