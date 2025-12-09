import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# ===== 設定 =====
input_folder = r"C:\Users\hrsyn\Desktop\PyT\emg_data_multiprocess"
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
target_ch = "CH3"

# ylimをファイル名ごとに指定（必要なものだけ書く）
ylim_settings = {
    "2emg_rms.csv": (0, 100),
    "3emg_ema.csv": (0, 100),
    # 指定がないファイルはデフォルトで (-150, 150) を使う
}

# ===== グラフ描画 =====
plt.figure(figsize=(10, len(csv_files) * 3))

for idx, file_path in enumerate(csv_files, start=1):
    df = pd.read_csv(file_path)
    time = df["Timestamp"] - df["Timestamp"].iloc[0]
    fname = os.path.basename(file_path)

    ax = plt.subplot(len(csv_files), 1, idx)
    ax.plot(time, df[target_ch], label=fname)
    ax.set_ylabel("Amplitude")

    # ylimをファイルごとに切り替え
    if fname in ylim_settings:
        ax.set_ylim(ylim_settings[fname])
    else:
        ax.set_ylim(-150, 150)  # デフォルト

    ax.set_title(f"{target_ch} - {fname}")
    ax.grid(True)

# plt.suptitle(f"Extracted Channel: {target_ch}", fontsize=18, y=0.99)
plt.subplots_adjust(hspace=0.5, top=0.95, bottom=0.05)
plt.show()
