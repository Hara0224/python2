import pandas as pd
import numpy as np
from collections import deque

# ========= パラメータ =========
RMS_WIN_MS = 20  # RMS 窓幅 [ms]
ALPHA = 0.2  # EMA 係数
THRESHOLD = 4.0  # Zスコア閾値
REFRACTORY = 0.1  # 不応期 [s]
PEAK_WIN = 0.08  # ピーク検出窓 [s]
CALIB_DURATION = 0.25  # キャリブ時間 [s]

# ========= データ読み込み =========
# CSV形式: Timestamp,CH1,CH2,...,CH8
df = pd.read_csv(r"C:\Users\AZUKI\Desktop\python\EMG2\20250729\emg_data_11_trim\emgraw_1_20250729_130655.csv")
time = df["Timestamp"].values.astype(float)
emg_raw = df.iloc[:, 1:].values  # CH1〜CH8
n_samples, n_channels = emg_raw.shape

# ========= 時間差を計算 =========
dt = np.median(np.diff(time))  # サンプリング間隔 [s]
FS = 1.0 / dt  # サンプリング周波数 [Hz]
N_RMS = max(3, int(FS * RMS_WIN_MS / 1000))

print(f"推定サンプリング周波数: {FS:.2f} Hz")

# ========= 前処理 =========
rms_buf = [deque(maxlen=N_RMS) for _ in range(n_channels)]
ema = np.zeros(n_channels)
rms_all, ema_all, z_all, trig_all = [], [], [], []

# ========= キャリブレーション =========
calib_end_time = time[0] + CALIB_DURATION
cal_data = []

for i in range(n_samples):
    if time[i] > calib_end_time:
        break
    for ch in range(n_channels):
        rms_buf[ch].append(emg_raw[i, ch])
        rms_val = np.sqrt(np.mean(np.square(rms_buf[ch])))
        ema[ch] = ALPHA * rms_val + (1 - ALPHA) * ema[ch]
    cal_data.append(ema.copy())

cal_data = np.array(cal_data)
mu = np.mean(cal_data, axis=0)
sigma = np.std(cal_data, axis=0, ddof=1) + 1e-6

# ========= メインループ =========
last_trigger_time = -1
in_trigger = False
trigger_ch = [1, 2, 5, 6]
peak_val = 0

for i in range(n_samples):
    t = time[i]
    z_vals = np.zeros(n_channels)

    for ch in range(n_channels):
        rms_buf[ch].append(emg_raw[i, ch])
        rms_val = np.sqrt(np.mean(np.square(rms_buf[ch])))
        ema[ch] = ALPHA * rms_val + (1 - ALPHA) * ema[ch]
        z_vals[ch] = (ema[ch] - mu[ch]) / sigma[ch]

    # トリガ検出
    if not in_trigger and (t - last_trigger_time) > REFRACTORY:
        if np.any(z_vals > THRESHOLD):
            trigger_ch = np.argmax(z_vals)
            peak_val = z_vals[trigger_ch]
            in_trigger = True
            trigger_time = t
            last_trigger_time = t
    elif in_trigger:
        if z_vals[trigger_ch] > peak_val:
            peak_val = z_vals[trigger_ch]
        if (t - trigger_time) > PEAK_WIN:
            in_trigger = False  # トリガ終了

    # 保存（Timestamp のみ）
    rms_all.append([t] + [np.sqrt(np.mean(np.square(rms_buf[ch]))) for ch in range(n_channels)])
    ema_all.append([t] + ema.tolist())
    z_all.append([t] + z_vals.tolist())
    trig_all.append(
        [
            t,
            int(in_trigger),
            trigger_ch if in_trigger else -1,
            peak_val if in_trigger else 0,
        ]
    )

# ========= CSV 出力 =========
col_names = ["Timestamp"] + [f"CH{c+1}" for c in range(n_channels)]

pd.DataFrame(rms_all, columns=col_names).to_csv("emg_rms.csv", index=False)
pd.DataFrame(ema_all, columns=col_names).to_csv("emg_ema.csv", index=False)
pd.DataFrame(z_all, columns=col_names).to_csv("emg_zscore.csv", index=False)
pd.DataFrame(trig_all, columns=["Timestamp", "TriggerFlag", "TriggerCH", "PeakVal"]).to_csv("emg_trigger.csv", index=False)

print("処理完了：emg_rms.csv, emg_ema.csv, emg_zscore.csv, emg_trigger.csv を保存しました。")
