import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import os
import glob

# === 1. 設定パラメータ (V12準拠) ===
SAVE_DIR = r"C:\Users\hrsyn\Desktop\PyT\emg_data_raw"
CHANNELS = [2, 3, 6, 7]
RAW_FS = 200
WINDOW_MS = 50
WINDOW_SAMPLES = int(RAW_FS * WINDOW_MS / 1000)  # 10 サンプル

# V12学習時に計算された値を使用
# T_DELTA (ΔRMS閾値)
T_DELTA = 19.7537
# T_ONSET (RMS静的閾値 - 比較のためV7/V9の値を使用)
T_ONSET_STATIC = 35.6482
DC_OFFSET = np.array([-0.57694795, -0.56773359, -0.63077918, -0.47289579])


# === 2. 特徴量計算関数 ===
def calculate_rms(window_data):
    """ウィンドウデータからRMSを計算 (4チャンネル分)"""
    return np.sqrt(np.mean(window_data**2, axis=0))


# === 3. データ処理とプロット関数 ===
def plot_emg_delta_onset_analysis(sample_label="radial_dev", num_trials=2):
    """RMSとDelta RMSを比較し、T_DeltaによるOnset検出を可視化する"""

    file_list = glob.glob(os.path.join(SAVE_DIR, f"emgraw_{sample_label}_*.csv"))

    if not file_list:
        print(f"❌ ラベル '{sample_label}' の生データファイルが見つかりませんでした。")
        return

    # 2行 (RMS/Delta RMS) × num_trials列のサブプロットを作成
    fig, axes = plt.subplots(
        num_trials, 1, figsize=(12, 5 * num_trials), sharex=True, squeeze=False
    )
    fig.suptitle(f"V12 Delta RMS Onset比較 ({sample_label} - Window 50ms)", fontsize=16)

    for i, file_path in enumerate(file_list[:num_trials]):
        df = pd.read_csv(file_path)

        emg_cols = [f"CH{c}" for c in CHANNELS]
        emg_data = df[emg_cols].values
        selected_emg = emg_data - DC_OFFSET

        rms_timeseries = []
        delta_rms_timeseries = []
        time_points = []
        previous_rms = None

        step = WINDOW_SAMPLES  # 50msステップ (10サンプル)

        # RMS/Delta RMS時系列の計算
        for j in range(0, len(selected_emg) - WINDOW_SAMPLES + 1, step):
            window = selected_emg[j : j + WINDOW_SAMPLES]
            current_rms = calculate_rms(window)

            # Delta RMS計算
            if previous_rms is None:
                delta_rms_max = 0.0
            else:
                # 立ち上がりを検出するため、正の最大値 (current_rms - previous_rms) を取得
                delta_rms_max = np.max(current_rms - previous_rms)

            rms_timeseries.append(np.max(current_rms))
            delta_rms_timeseries.append(delta_rms_max)
            time_points.append((j + WINDOW_SAMPLES / 2) / RAW_FS)
            previous_rms = current_rms

        # === グラフ描画 ===
        # 1. RMS時系列
        ax1 = axes[i, 0]
        ax1.plot(
            time_points,
            rms_timeseries,
            label="Max RMS",
            color="blue",
            marker=".",
            linestyle="-",
        )
        ax1.axhline(
            T_ONSET_STATIC,
            color="red",
            linestyle="--",
            label=f"Static RMS Threshold ({T_ONSET_STATIC:.1f})",
        )
        ax1.set_ylabel("RMS Value", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.set_title(f"試行 {i+1} ({os.path.basename(file_path)})")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. Delta RMS時系列 (第二Y軸)
        ax2 = ax1.twinx()
        ax2.plot(
            time_points,
            delta_rms_timeseries,
            label="Max Delta RMS (Rate of Change)",
            color="green",
            marker="x",
            linestyle="-",
        )
        ax2.axhline(
            T_DELTA,
            color="black",
            linestyle="-",
            label=f"Delta RMS Threshold ({T_DELTA:.1f})",
        )
        ax2.set_ylabel("Delta RMS Value", color="green")
        ax2.tick_params(axis="y", labelcolor="green")

        # 3. Delta RMSによるOnset検出点のマーク (V12システムの真のOnset)
        delta_rms_array = np.array(delta_rms_timeseries)
        onset_indices = np.where(delta_rms_array >= T_DELTA)[0]

        if len(onset_indices) > 0:
            first_onset_idx = onset_indices[0]
            onset_time = time_points[first_onset_idx]

            # 検出タイミングの強調 (すべてのグラフで共通)
            ax1.axvline(
                onset_time,
                color="purple",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="V12 Onset Time",
            )
            ax1.plot(
                [],
                [],
                color="purple",
                linestyle="--",
                linewidth=2,
                label="V12 Onset Time",
            )  # 凡例用
            ax2.plot(
                onset_time,
                delta_rms_array[first_onset_idx],
                "ro",
                markersize=10,
                label="Delta Onset Point",
            )

        # 凡例の統合
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        if i == num_trials - 1:
            ax1.set_xlabel("Time (seconds)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# === 4. 実行 ===
if __name__ == "__main__":
    # radial_dev（撓屈）の試行2つをプロット
    plot_emg_delta_onset_analysis(sample_label="radial_dev", num_trials=4)
