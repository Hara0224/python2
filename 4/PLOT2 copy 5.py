import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import japanize_matplotlib
import glob

# === 1. 設定パラメータ (V11準拠) ===
SAVE_DIR = r"H:\マイドライブ\GooglePython\EMG\SVM2"
CHANNELS = [2, 3, 6, 7]  # 使用するチャンネル
CHANNEL_LABELS = [f"CH{c}" for c in CHANNELS]  # プロット用ラベル
RAW_FS = 200
WINDOW_MS = 50
WINDOW_SAMPLES = int(RAW_FS * WINDOW_MS / 1000)  # 10 サンプル (窓サイズ)

# ステップサイズを定義 (50%重複 = 25msステップ)
STEP_MS = 5
STEP_SAMPLES = int(RAW_FS * STEP_MS / 1000)  # 5 サンプル (ステップサイズ)

# V11学習時に計算された値を使用
T_ONSET = 31.9180  # <<<< V11のT_ONSET (K=3.0) >>>>
DC_OFFSET = np.array([-0.57694795, -0.56773359, -0.63077918, -0.47289579])


# === 2. RMS計算関数 ===
def calculate_rms(window_data):
    """ウィンドウデータからRMSを計算 (4チャンネル分)"""
    return np.sqrt(np.mean(window_data**2, axis=0))


# === 3. データ処理とプロット関数 ===
def plot_emg_onset_analysis(sample_label, num_trials=3):
    """指定されたラベルの試行について、各チャンネルのRMSを、チャンネルごとに縦にまとめてプロットする"""

    file_list = glob.glob(os.path.join(SAVE_DIR, f"emgraw_{sample_label}_*.csv"))

    if not file_list:
        print(f"❌ ラベル '{sample_label}' の生データファイルが見つかりませんでした。")
        return

    selected_files = file_list[:num_trials]
    num_trials_actual = len(selected_files)

    if num_trials_actual == 0:
        print("⚠️ プロットするファイルがありませんでした。")
        return

    num_channels = len(CHANNELS)
    fig, axes = plt.subplots(
        num_channels,
        num_trials_actual,
        figsize=(4 * num_trials_actual, 4 * num_channels),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    fig.suptitle(
        f"V11 Onset立ち上がり分析 - チャンネル別比較 ({sample_label} - {WINDOW_MS}ms Window / {STEP_MS}ms Step)",
        fontsize=16,
    )

    if sample_label == "radial_dev":
        onset_ch_indices = [0, 1]  # CH2, CH3
        onset_ch_names = "CH2, CH3"
    elif sample_label == "ulnar_dev":
        onset_ch_indices = [2, 3]  # CH6, CH7
        onset_ch_names = "CH6, CH7"
    else:
        onset_ch_indices = list(range(num_channels))
        onset_ch_names = "全4ch"

    all_trial_data = []

    for file_path in selected_files:
        df = pd.read_csv(file_path)

        emg_cols = [f"CH{c}" for c in CHANNELS]
        emg_data = df[emg_cols].values

        # DCオフセット除去
        selected_emg = emg_data - DC_OFFSET

        rms_timeseries_all_ch = []
        time_points = []

        step = STEP_SAMPLES

        for j in range(0, len(selected_emg) - WINDOW_SAMPLES + 1, step):
            window = selected_emg[j : j + WINDOW_SAMPLES]
            rms_vector = calculate_rms(window)
            rms_timeseries_all_ch.append(rms_vector)
            time_points.append((j + WINDOW_SAMPLES / 2) / RAW_FS)

        rms_array_all_ch = np.array(rms_timeseries_all_ch)

        max_rms_per_window = np.max(rms_array_all_ch[:, onset_ch_indices], axis=1)

        all_trial_data.append(
            {
                "rms_array": rms_array_all_ch,
                "time_points": time_points,
                "max_rms": max_rms_per_window,
                "filename": os.path.basename(file_path),
                "onset_ch_names": onset_ch_names,
            }
        )

    for ch_idx in range(num_channels):
        ch_label = CHANNEL_LABELS[ch_idx]

        for i, trial_data in enumerate(all_trial_data):
            ax = axes[ch_idx, i]

            rms_timeseries = trial_data["rms_array"][:, ch_idx]
            time_points = trial_data["time_points"]
            max_rms_per_window = trial_data["max_rms"]

            ax.plot(
                time_points,
                rms_timeseries,
                label=f"RMS ({ch_label})",
                color=f"C{ch_idx}",
                marker="o",
                linestyle="",
                markersize=3,
            )

            onset_indices = np.where(max_rms_per_window >= T_ONSET)[0]

            if len(onset_indices) > 0:
                first_onset_idx = onset_indices[0]
                onset_time = time_points[first_onset_idx]

                # ------------------------------------------------------------------
                # 変更点: Onset Timeの線の色を "red" に変更
                ax.axvline(
                    onset_time,
                    color="red",  # ここを "red" に変更しました
                    linestyle=":",
                    alpha=0.6,
                    label=f"Onset Time ({onset_time:.3f} s)",
                )
                # ------------------------------------------------------------------

            if ch_idx == 0:
                ax.set_title(
                    f"試行 {i+1}\n({trial_data['filename'][:15]}...)", fontsize=10
                )

            if i == 0:
                ax.set_ylabel(f"{ch_label}\nRMS Value", fontsize=10)

                ax.text(
                    0.02,
                    0.98,
                    f"T_Onset (Max {trial_data['onset_ch_names']}): {T_ONSET:.2f}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5, lw=0),
                    fontsize=8,
                )

            if ch_idx == num_channels - 1:
                ax.set_xlabel("Time (seconds)", fontsize=10)

            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# === 4. 実行 ===
if __name__ == "__main__":
    plot_emg_onset_analysis(sample_label="radial_dev", num_trials=3)
    plot_emg_onset_analysis(sample_label="ulnar_dev", num_trials=3)
