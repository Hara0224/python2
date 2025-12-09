import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# === 設定 ===
data_dir = "./emg_data_raw_air"  # CSV保存ディレクトリ
window_size = 20             # 50サンプル（約250ms）
stride = 10                  # ずらし幅（半分オーバーラップ）
use_channels = [f"CH{i+1}" for i in range(8)]

# === データ読み込み & 前処理 ===
X = []
y = []

for filename in os.listdir(data_dir):
    if not filename.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(data_dir, filename))
    label = df["Label"].iloc[0]
    emg = df[use_channels].values  # shape: (T, 8)

    # ウィンドウ分割
    for i in range(0, len(emg) - window_size + 1, stride):
        window = emg[i:i+window_size]
        X.append(window)
        y.append(label)

# numpy配列に変換
X = np.array(X)                      # shape: (N, 50, 8)
X = np.transpose(X, (0, 2, 1))       # shape: (N, 8, 50)
X = X / 128.0                        # [-1, 1] に正規化（±128のレンジ前提）

# ラベルを数値に変換
le = LabelEncoder()
y = le.fit_transform(y)              # u=2 "n"→1, "down"→0 など
print("ラベル → 数値対応:", dict(zip(le.classes_, le.transform(le.classes_))))
# 学習・テストに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


np.savez("emg_dataset_air.npz",
         X_train=X_train,
         X_test=X_test,
         y_train=y_train,
         y_test=y_test)
print("✅ emg_dataset_air.npz に保存しました。")