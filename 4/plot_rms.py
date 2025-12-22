import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

input_folder = r"C:\Users\hrsyn\Desktop\masterPY\emg_data_13"  # 整流化されたCSVが入っているフォルダ
window_size = 50  # RMSのウィンドウサイズ

csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

# 表示したいチャンネルを指定（例: CH1, CH3, CH5）
selected_channels = [2, 3, 5, 6]

for file_path in csv_files:
    df = pd.read_csv(file_path)
    time = df["Timestamp"] - df["Timestamp"][0]

    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # グラフ描画
    plt.figure(figsize=(10, 5 * len(selected_channels)))
    for idx, ch in enumerate(selected_channels, start=1):
        # RMS計算 (Rolling Root Mean Square)
        rms = (df[f"CH{ch}"] ** 2).rolling(window=window_size).mean() ** 0.5
        
        plt.subplot(len(selected_channels), 1, idx)
        plt.plot(time, rms, label=f"CH{ch} (RMS)")
        plt.title(f"Channel {ch}", fontsize=20)
        plt.ylabel("Amplitude (RMS)")
        plt.ylim(0, 150)
        plt.grid(True)
        plt.legend()

    plt.suptitle(f"{file_name} (RMS window={window_size})", fontsize=18, y=0.96)
    plt.xlabel("Time (s)", y=0.05)
    plt.subplots_adjust(hspace=0.6, top=0.9, bottom=0.05)

    plt.show()
