import pandas as pd
import matplotlib.pyplot as plt
import os

# === パス設定 ===
emg_csv_path = r"C:\Users\AZUKI\Desktop\python\EMG2\20250813\emg_data_rms1\emg_rms_1_20250812_171525.csv"  # ← 1つのEMGファイルを指定
angle_csv_path = r"C:\Users\AZUKI\Desktop\python\EMG2\20250813\img_data_rms1\1_20250812_171525\results.csv"  # 角度CSV

# === 角度CSVの読み込み ===
angle_df = pd.read_csv(angle_csv_path)
angle_dict = dict(zip(angle_df["filename"].astype(str), angle_df["angle_deg"]))

# === EMGデータの読み込み ===
df = pd.read_csv(emg_csv_path)
time = df["Timestamp"] - df["Timestamp"][0]
base_name = os.path.splitext(os.path.basename(emg_csv_path))[0]
angle = angle_dict.get(base_name, None)
angle_str = f"{angle:.1f}°" if angle is not None else "N/A"

# === 角度グラフ用のソート済みデータ ===
angle_df["filename"] = angle_df["filename"].astype(str)
angle_df = angle_df.sort_values(by="filename", key=lambda x: x.astype(int))
x_labels = angle_df["filename"].tolist()
y_values = angle_df["angle_deg"].tolist()

# === プロット（9段） ===
fig, axs = plt.subplots(9, 1, figsize=(10, 32), sharex=False)

# 8ch EMGデータの描画
for i in range(8):
    axs[i].plot(time, df[f"CH{i+1}"], label=f"CH{i+1}")
    axs[i].set_ylabel("Amplitude")
    axs[i].set_ylim(-10, 1000)
    axs[i].set_title(f"Channel {i+1}", y=0.93)
    axs[i].grid(True)
    # axs[i].set_xlabel('Time (s)')
    axs[i].set_xlim(time.min(), time.max())
    if i != 7:
        axs[i].tick_params(labelbottom=False)
axs[7].set_xlabel("Time (s)", fontsize=14)

frame_rate = 30  # 30Hz

# === 角度CSV処理 ===
angle_df["filename"] = angle_df["filename"].astype(float).astype(int)
angle_df = angle_df.drop_duplicates(subset="filename")
angle_df = angle_df.sort_values(by="filename")

x_labels = angle_df["filename"].astype(str).tolist()
y_values = angle_df["angle_deg"].tolist()
x_time = angle_df["filename"] / frame_rate  # フレーム番号 → 秒に変換

# === 角度 vs 時間 のプロット（9段目）
axs[8].plot(x_time, y_values, marker="o", linestyle="-")
# axs[8].set_xlabel('Time (s)')
axs[8].set_ylabel("Angle (deg)")
axs[8].set_xticks(x_time)
axs[8].set_xticklabels(x_labels, rotation=90)
axs[8].grid(True)
axs[8].set_xlim([x_time.min(), x_time.max()])

# タイトルと余白調整
fig.suptitle(f"8ch EMG: {base_name} ", fontsize=15, y=0.99)
plt.subplots_adjust(hspace=0.8, top=0.8, bottom=0.05)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
