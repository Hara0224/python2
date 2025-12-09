import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

# === パラメータ ===
input_folder = r'C:\Users\hrsyn\Desktop\Python\EMG123\emg_data_11trim_abs'       # 入力CSVフォルダ
output_folder = 'emg_data11_bandpas/'  # 出力CSV保存先
os.makedirs(output_folder, exist_ok=True)

lowcut = 20.0      # 下限周波数 (Hz)
highcut = 99.0    # 上限周波数 (Hz)fs/2の値まで
fs = 200.0         # サンプリング周波数 (Hz) ← 実際の値に合わせて変更

# === バンドパスフィルタ関数 ===
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# === フォルダ内のCSV処理 ===
csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

for file in csv_files:
    df = pd.read_csv(file)

    # 前提：1列目=タイムスタンプ, 2～9列目=EMG(8ch), 最終列=ラベル
    timestamp = df.iloc[:, 0]
    emg_data = df.iloc[:, 1:9].values
    label_col = df.iloc[:, -1]

    # バンドパスフィルタ適用
    filtered_emg = bandpass_filter(emg_data, lowcut, highcut, fs)

    # 新しいデータフレーム作成
    filtered_df = pd.DataFrame(filtered_emg, columns=[f'CH{i+1}' for i in range(8)])
    filtered_df.insert(0, 'Timestamp', timestamp)
    filtered_df['label'] = label_col

    # 保存
    base_name = os.path.basename(file)
    output_path = os.path.join(output_folder, f'emgband_{base_name}')
    filtered_df.to_csv(output_path, index=False)

    print(f"✅ 保存完了: {output_path}")
