import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import japanize_matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# === 設定パラメータ (学習コードと一致させる) ===
MODEL_SAVE_PATH = "svm_hybrid_v12_delta.joblib"
SCALER_SAVE_PATH = "scaler_hybrid_v12_delta.joblib"
FEATURE_DATA_PATH = "svm_input_features_hybrid_v12.csv"
CHANNELS = [2, 3, 6, 7]
FEATURE_NAMES = [f"RMS_CH{c}" for c in CHANNELS] + [f"DeltaRMS_CH{c}" for c in CHANNELS]
STEP_SIZE = 0.05  # 2D決定境界の計算負荷軽減のため0.05を使用

# === 1. モデルとデータのロード ===
print("--- 1. モデルとデータのロード ---")
try:
    df_features = pd.read_csv(FEATURE_DATA_PATH)
    scaler = joblib.load(SCALER_SAVE_PATH)
    svm = joblib.load(MODEL_SAVE_PATH)
except FileNotFoundError as e:
    print(f"❌ 必要なファイルが見つかりません。学習コードを実行してファイルを生成してください: {e}")
    exit()

# ラベルを数値にエンコード
le = LabelEncoder()
y_all = df_features["Label"].values
y_numeric = le.fit_transform(y_all)
label_names = le.classes_  # ['Movement', 'rest']

X_all = df_features[FEATURE_NAMES].values
X_scaled = scaler.transform(X_all)


# === 2. グラフ化関数 ===


def plot_pca_decision_boundary(X_scaled, y_numeric, svm, scaler, pca, label_names, title_suffix=""):
    """PCA 2D空間におけるデータ分布とSVMの決定境界をプロットする"""
    print("--- 2-1. PCAと決定境界の計算開始 ---")

    # PCAの実行（8D -> 2Dへ圧縮）
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"✅ PCA実行完了。寄与率 (2D): {explained_variance:.4f}")

    # グリッド生成
    x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
    y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, STEP_SIZE), np.arange(y_min, y_max, STEP_SIZE))

    # 決定境界の予測
    grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
    X_8d_restored = pca.inverse_transform(grid_points_2d)
    X_8d_scaled = scaler.transform(X_8d_restored)
    Z = svm.predict(X_8d_scaled)

    # 予測結果の整形
    Z_numeric = le.transform(Z)
    Z_mesh = Z_numeric.reshape(xx.shape)

    # プロット
    plt.figure(figsize=(10, 8))
    cmap_scatter = plt.cm.get_cmap("coolwarm", len(label_names))

    # 決定境界の塗りつぶし
    plt.contourf(xx, yy, Z_mesh, alpha=0.3, cmap=cmap_scatter)

    # 決定境界の線描画
    plt.contour(
        xx,
        yy,
        Z_mesh,
        levels=len(label_names) - 1,
        colors="k",
        linewidths=1.5,
        linestyles="solid",
    )

    # データ点のプロット
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y_numeric,
        cmap=cmap_scatter,
        s=20,
        edgecolors="k",
        alpha=0.8,
    )

    # カラーバーと凡例の設定
    cbar = plt.colorbar(scatter, ticks=np.arange(len(label_names)), label="Class")
    cbar.ax.set_yticklabels(label_names)

    plt.title(f"SVM 決定境界 (PCA 2D) {title_suffix}\nPCA寄与率: {explained_variance:.4f}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
    print("✅ PCA 決定境界プロット完了。")


def plot_confusion_matrix_and_report(y_test, y_pred, label_names, title):
    """混同行列とClassification Reportのヒートマップをプロットする"""
    print(f"\n--- 2-2. {title}の可視化開始 ---")

    # 混同行列
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.xlabel("予測ラベル")
    plt.ylabel("真のラベル")
    plt.title(f"{title} 混同行列")
    plt.show()

    print("✅ 混同行列プロット完了。")
    # classification_reportのテキスト出力は学習コードで行われているため、ここでは省略


# === 3. グラフの実行 ===


def plot_main():
    # A. 決定境界のプロット
    # モデルの学習にはテストデータが必要なため、ダミーのPCAオブジェクトを使用
    plot_pca_decision_boundary(X_scaled, y_numeric, svm, scaler, PCA(n_components=2), label_names, "(全データ)")

    # B. 混同行列のプロットには、テスト結果のデータが必要
    # 学習コードからテストデータと予測結果を再生成する必要がある

    from sklearn.model_selection import train_test_split

    # ラベルを再生成（学習コードから抽出）
    X = df_features[FEATURE_NAMES].values
    y = df_features["Label"].values
    weights = df_features["SampleWeight"].values

    X_train, X_test, y_train, y_test, _, _ = train_test_split(X, y, weights, test_size=0.2, random_state=42, stratify=y)

    # スケーリング
    X_test_scaled = scaler.transform(X_test)

    # 再度予測
    y_pred = svm.predict(X_test_scaled)

    # 混同行列のプロット
    plot_confusion_matrix_and_report(y_test, y_pred, label_names, "SVM Onset Model (テストデータ)")


if __name__ == "__main__":
    plot_main()
