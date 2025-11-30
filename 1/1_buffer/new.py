import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import matplotlib.ticker as ticker

# === WAVファイルを読み込み ===
filename = r"C:\Users\AZUKI\Desktop\python\EMG2\zikkenn1.wav"
sample_rate, data = wavfile.read(filename)

# ステレオ（2ch）の場合は片方だけ
if data.ndim == 2:
    data = data[:, 0]

# 時間軸を作成
time = np.linspace(0, len(data) / sample_rate, num=len(data))

# === 波形をプロット ===
fig, ax = plt.subplots(figsize=(12.8, 2.0))  # インチ指定 (100dpiで 1280x200px 相当)
ax.plot(time, data, linewidth=0.7)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Amplitude")
# ax.set_title(f"Waveform of {filename}")
ax.grid(True)

ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))  # 1秒ごと
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))  # 0.1秒ごと

# === 画像保存 (1280x200 px) ===
plt.savefig("waveform.png", dpi=100, bbox_inches="tight", pad_inches=0.1)
plt.close()
