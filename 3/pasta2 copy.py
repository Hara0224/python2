import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# 1. データ定義
# ==========================================
# 行：正解データ（上段：弱、下段：強）
# 列：予測データ（左列：弱、右列：強）
cm = np.array([[19, 1], [4, 16]])  # 正解が「弱」のとき: [弱と判定(正), 強と判定(誤)]  # 正解が「強」のとき: [弱と判定(誤), 強と判定(正)]

# ラベル定義
labels = ["弱 (Weak)", "強 (Strong)"]

# ==========================================
# 2. 日本語フォントの設定
# ==========================================
# Windows標準の「MSゴシック」を指定します
plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["font.size"] = 12

# ==========================================
# 3. グラフ描画
# ==========================================
plt.figure(figsize=(6, 5))

# ヒートマップ作成
# cmap='Blues': 青色の濃淡（論文などできれいに見えます）
# annot=True: 数字を表示
# fmt='d': 整数で表示
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, annot_kws={"size": 20}, cbar=False, square=True, linewidths=1, linecolor="black")

# 軸ラベルとタイトル
plt.xlabel("予測ラベル ", fontsize=14, fontweight="bold")
plt.ylabel("正解ラベル ", fontsize=14, fontweight="bold")
# plt.title("強弱判定の混合行列", fontsize=16, pad=20)

# ==========================================
# 4. 保存と表示
# ==========================================
plt.tight_layout()
plt.savefig("confusion_matrix_jp.png", dpi=300)
print("画像 'confusion_matrix_jp.png' を保存しました。")
plt.show()
