import pandas as pd
import matplotlib.pyplot as plt

# === パスの設定 ===
csv_path = r"C:\Users\AZUKI\Desktop\python\EMG2\20250813\emg_data_rms1\emg_rms_1_20250812_171525.csv"  # ←ここを実ファイルパスに
timestamp_col = "Timestamp"  # タイムスタンプの列名（例：1753762159.131577 など）
channel_cols = [f"CH{i}" for i in range(1, 9)]  # CH1～CH8

# === CSV読み込み ===
df = pd.read_csv(csv_path)

# 確認（念のため）
print("CSVの列名：", df.columns.tolist())

# === 時間を0秒始まりに変換 ===
df["time_sec"] = df[timestamp_col] - df[timestamp_col].iloc[0]

# === グラフ描画（クリックで平均範囲指定） ===
fig, axes = plt.subplots(8, 1, figsize=(12, 10), sharex=True)
clicks = []


def onclick(event):
    if event.inaxes:
        clicks.append(event.xdata)
        print(f"クリック: {event.xdata:.3f} 秒")

        if len(clicks) == 2:
            t_start, t_end = sorted(clicks)
            mask = (df["time_sec"] >= t_start) & (df["time_sec"] <= t_end)
            selected = df.loc[mask, channel_cols]
            means = selected.mean()
            print(f"\n📊 {t_start:.3f}〜{t_end:.3f} 秒のチャンネル平均:")
            for ch, val in means.items():
                print(f"  {ch}: {val:.3f}")
            clicks.clear()  # リセット


# 各チャンネルをプロット
for i, ch in enumerate(channel_cols):
    axes[i].plot(df["time_sec"], df[ch], label=ch)
    axes[i].legend(loc="upper right")
    axes[i].set_ylabel(ch)

axes[-1].set_xlabel("Time (sec)")
fig.suptitle("8chaverage", fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.95)

# クリックイベントの登録
fig.canvas.mpl_connect("button_press_event", onclick)

plt.show()
