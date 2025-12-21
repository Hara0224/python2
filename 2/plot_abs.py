import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

input_folder = r"C:\Users\hrsyn\Desktop\DATAforPython\20250729\emg_data_raw11"  # 整流化されたCSVが入っているフォルダ

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
        plt.subplot(len(selected_channels), 1, idx)
        plt.plot(time, df[f"CH{ch}"], label=f"CH{ch}")
        plt.title(f"Channel {ch}", fontsize=20)
        plt.ylabel("Amplitude")
        plt.ylim(-150, 150)
        plt.grid(True)
        plt.legend()

    # plt.suptitle(f"Channels - {file_name}", fontsize=18, y=0.96)
    plt.xlabel("Time (s)", y=0.05)
    plt.subplots_adjust(hspace=0.6, top=0.9, bottom=0.05)

    plt.show()
