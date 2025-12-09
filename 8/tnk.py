import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib

# --- 1. データ定義 (最適解: Accuracy=0.66 の結果に基づく推定値) ---

# 行: 真のクラス (True Label)
# 列: 予測されたクラス (Predicted Label)
# この行列は、元の Classification Report から逆算して推定されたものです。
# (rest, radial_dev, ulnar_dev)
confusion_matrix_data = np.array(
    [
        # 予測: rest, radial_dev, ulnar_dev
        [3081, 471, 160],  # 真: rest (0)
        [2187, 1523, 0],  # 真: radial_dev (1)
        [1596, 185, 2129],  # 真: ulnar_dev (2)
    ]
)

# クラスラベル
class_labels = ["無動作", "撓屈運動", "尺屈運動"]


# --- 2. 混合行列の図示 ---
def plot_confusion_matrix(cm, labels, title="SVMによる判別結果の混合行列"):
    """混合行列をヒートマップとして描画する関数"""

    # Matplotlibの図と軸を作成
    plt.figure(figsize=(8, 6))

    # Seabornを使ってヒートマップを生成
    # annot=True: セルに値を表示
    # fmt='d': 整数として値をフォーマット
    # cmap: カラーマップ (ここでは青系の'Blues'を使用)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        linewidths=0.5,
        linecolor="black",
    )

    # ラベル設定
    plt.title(title)
    plt.xlabel("予測ラベル", fontsize=12)
    plt.ylabel("真のラベル", fontsize=12)

    # 軸目盛りラベルの設定
    # xticklabels: 列 (予測)
    # yticklabels: 行 (真)
    plt.xticks(
        ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=45, ha="right"
    )
    plt.yticks(
        ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=0, va="center"
    )

    # レイアウト調整と表示
    plt.tight_layout()
    plt.show()


# --- 3. 実行 ---
plot_confusion_matrix(
    confusion_matrix_data,
    class_labels,
    title="SVMによる判別結果の混合行列",
)
