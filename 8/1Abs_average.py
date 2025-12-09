import pandas as pd
import matplotlib.pyplot as plt

# CSVファイル読み込み
df = pd.read_csv(r'C:\Users\hrsyn\Desktop\Python\emg_data_high\emgraw_2up_20250721_113005.csv')

# 時間軸（0秒からの経過時間）
time = df['Timestamp'] - df['Timestamp'][0]

# CH1〜CH8を絶対値に変換
for i in range(1, 9):
    df[f'CH{i}'] = df[f'CH{i}'].abs()

# 各チャンネルの平均値を計算して表示
print("=== 各チャンネルの絶対値平均 ===")
for i in range(1, 9):
    mean_val = df[f'CH{i}'].mean()
    print(f"CH{i}: {mean_val:.2f}")

# グラフ描画
plt.figure(figsize=(15, 10))
for i in range(1, 9):
    plt.subplot(4, 2, i)
    plt.plot(time, df[f'CH{i}'], label=f'CH{i}')
    plt.title(f'Channel {i}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (abs)')
    plt.ylim(0, 150)  # 絶対値なので下限は0
    plt.grid(True)

plt.suptitle("8ch_data (Absolute Value)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

