import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import japanize_matplotlib
from sklearn.svm import SVC
import joblib
import os

# === 設定パスとファイル名 ===
MODEL_PATH = "svm_onset_model_rms_tuned_v6.joblib"
SCALER_PATH = "scaler_onset_rms_tuned_v6.joblib"
# 特徴量データセットファイル名 (V4生成時にファイル名を変更していない場合)
FEATURE_DATA_PATH = r"C:\Users\hrsyn\Desktop\Python\svm_input_features_rms_v5.csv"
CHANNELS = [2, 3, 6, 7]
FEATURE_NAMES = [f"RMS_CH{c}" for c in CHANNELS] + [f"DeltaRMS_CH{c}" for c in CHANNELS]

# === 1. モデルとデータのロード ===
print("--- 1. モデルとデータのロード ---")

# モデル・スケーラーのロード
try:
    scaler = joblib.load(SCALER_PATH)
    svm = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"❌ モデル ({MODEL_PATH}) またはスケーラー ({SCALER_PATH}) ファイルが見つかりません。パスを確認してください。")
    exit()

# 特徴量データセットのロード
try:
    df_features = pd.read_csv(FEATURE_DATA_PATH)
    X_all = df_features[FEATURE_NAMES].values
    y_all = df_features["Label"].values
except FileNotFoundError:
    print(f"❌ 特徴量データセット ({FEATURE_DATA_PATH}) が見つかりません。")
    exit()

# ラベルを数値にエンコード
le = LabelEncoder()
y_numeric = le.fit_transform(y_all)
label_names = le.classes_  # ['radial_dev', 'rest', 'ulnar_dev']など


# === 2. データの準備とPCAによる次元削減 ===
print("--- 2. PCAによる次元削減 (8D -> 2D) ---")

# 特徴量の標準化 (学習時と同じスケーラーを使用)
X_scaled = scaler.transform(X_all)

# PCAの初期化と実行（8次元 -> 2次元へ圧縮）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"✅ PCA実行完了。寄与率: {np.sum(pca.explained_variance_ratio_):.4f}")  # 寄与率を確認


# === 3. 決定境界のプロット ===
print("--- 3. 決定境界の計算とプロット ---")

# プロットエリアのグリッドを生成
# グリッドの範囲をデータ点の範囲より少し広く設定
x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
# 解像度 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

# グリッドポイント (xx.ravel(), yy.ravel()) を作成
# 2次元のグリッドを8次元に戻し、スケーラーで標準化してからSVMで予測を行う
grid_points_2d = np.c_[xx.ravel(), yy.ravel()]

# PCA逆変換 -> 標準化の逆変換（厳密には不要だが、PCAの特徴量空間を標準化する前に戻す）
# 実際には、PCAを介して学習されたSVMは、標準化された2次元特徴量で予測する
# しかし、プロットのために元の8次元空間の境界を正確に模倣する必要がある。
# ここでは、PCA逆変換 -> スケーラー予測の順序で処理する。
X_8d_restored = pca.inverse_transform(grid_points_2d)
X_8d_scaled = scaler.transform(X_8d_restored)  # 再度標準化を適用

# SVMで予測
Z = svm.predict(X_8d_scaled)

# 予測結果のラベルを数値に戻す
Z_numeric = le.transform(Z)
Z_mesh = Z_numeric.reshape(xx.shape)

# プロット設定
plt.figure(figsize=(10, 8))
cmap = plt.cm.get_cmap("RdYlBu", len(label_names))  # クラス数に応じたカラーマップ

# 予測結果（決定境界）を塗りつぶし
plt.contourf(xx, yy, Z_mesh, alpha=0.7, cmap=cmap)

# 元のデータ点をプロット
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_numeric, cmap=cmap, s=20, edgecolors="k")

# カラーバーと凡例の設定
cbar = plt.colorbar(scatter, ticks=np.arange(len(label_names)), label="Class")
cbar.ax.set_yticklabels(label_names)

# タイトルとラベル
plt.title(f"SVM RBF Kernel 決定境界 (PCA 2D)\nPCA寄与率: {np.sum(pca.explained_variance_ratio_):.4f}")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

print("✅ 決定境界のプロットが完了しました。")
