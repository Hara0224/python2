import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import japanize_matplotlib
import glob

# === 1. 設定パラメータ (V11準拠) ===
SAVE_DIR = "./emg_data_raw/"
CHANNELS = [2, 3, 6, 7]
RAW_FS = 200
WINDOW_MS = 50
WINDOW_SAMPLES = int(RAW_FS * WINDOW_MS / 1000)  # 10 サンプル

# V11学習時に計算された値を使用
T_ONSET = 31.9180  # <<<< V11のT_ONSET (K=3.0) >>>>
DC_OFFSET = np.array([-0.57694795, -0.56773359, -0.63077918, -0.47289579])


# === 2. RMS計算関数 ===
def calculate_rms(window_data):
    """ウィンドウデータからRMSを計算 (4チャンネル分)"""
    return np.sqrt(np.mean(window_data**2, axis=0))


# === 3. データ処理とプロット関数 ===
def plot_emg_onset_analysis(sample_label="radial_dev", num_trials=3):
    """指定されたラベルの試行についてRMSとOnset閾値をプロットする"""

    file_list = glob.glob(os.path.join(SAVE_DIR, f"emgraw_{sample_label}_*.csv"))

    if not file_list:
        print(f"❌ ラベル '{sample_label}' の生データファイルが見つかりませんでした。")
        return

    fig, axes = plt.subplots(num_trials, 1, figsize=(12, 4 * num_trials), sharex=True, squeeze=False)
    fig.suptitle(
        f"V11 Onset立ち上がり分析 ({sample_label} - 50ms Window, T_onset={T_ONSET:.2f})",
        fontsize=16,
    )

    for i, file_path in enumerate(file_list[:num_trials]):
        df = pd.read_csv(file_path)

        emg_cols = [f"CH{c}" for c in CHANNELS]
        emg_data = df[emg_cols].values

        # DCオフセット除去
        selected_emg = emg_data - DC_OFFSET

        rms_timeseries = []
        time_points = []

        # RMSは50msウィンドウ/50msステップで計算
        step = WINDOW_SAMPLES

        for j in range(0, len(selected_emg) - WINDOW_SAMPLES + 1, step):
            window = selected_emg[j : j + WINDOW_SAMPLES]
            rms_vector = calculate_rms(window)
            rms_timeseries.append(np.max(rms_vector))
            time_points.append((j + WINDOW_SAMPLES / 2) / RAW_FS)

        ax = axes[i, 0]
        # 点のみのプロット (50msごとの離散特徴量)
        ax.plot(
            time_points,
            rms_timeseries,
            label="Max RMS (4ch)",
            color="blue",
            marker="o",
            linestyle="",
            markersize=4,
        )

        # Onset閾値の描画
        ax.axhline(
            T_ONSET,
            color="red",
            linestyle="--",
            label=f"Onset Threshold ({T_ONSET:.1f})",
        )

        # 閾値を超えた最初の点をマーク
        rms_array = np.array(rms_timeseries)
        onset_indices = np.where(rms_array >= T_ONSET)[0]
        if len(onset_indices) > 0:
            first_onset_idx = onset_indices[0]
            ax.axvline(
                time_points[first_onset_idx],
                color="green",
                linestyle=":",
                label="System Onset Time",
            )
            ax.plot(
                time_points[first_onset_idx],
                rms_array[first_onset_idx],
                "go",
                markersize=8,
            )

        ax.set_title(f"試行 {i+1}: {os.path.basename(file_path)}", fontsize=12)
        ax.set_ylabel("RMS Value", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.7)

    plt.xlabel("Time (seconds)", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# === 4. 実行 ===
if __name__ == "__main__":
    # radial_dev（撓屈）の試行3つをプロット
    plot_emg_onset_analysis(sample_label="radial_dev", num_trials=1)
