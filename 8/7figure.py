import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --------------------------
# モデル定義（学習時と同じ構造にする）
# --------------------------
class EMG_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # ← ラベルが3クラスの場合
        )
    def forward(self, x):
        return self.net(x)

# --------------------------
# データ読み込み
# --------------------------
data = np.load("emg_dataset_air.npz")

X_test = data["X_test"]  # shape: (N, 50, 8)
y_test = data["y_test"]

# Conv1dのため shape: (N, 8, 50) に変換
X_test = X_test.transpose(0, 1, 2)

X_tensor = torch.tensor(X_test, dtype=torch.float32)
y_tensor = torch.tensor(y_test, dtype=torch.long)

test_dataset = TensorDataset(X_tensor, y_tensor)
test_loader = DataLoader(test_dataset, batch_size=16)

# --------------------------
# モデル読み込み
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EMG_CNN().to(device)
model.load_state_dict(torch.load("cnn_emg_model_air.pth", map_location=device))
model.eval()

# --------------------------
# テスト評価
# --------------------------
criterion = nn.CrossEntropyLoss()
total_loss = 0
correct = 0
total = 0

all_preds = []
all_targets = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        outputs = model(xb)
        loss = criterion(outputs, yb)

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(yb.cpu().numpy())

# --------------------------
# 結果表示
# --------------------------
avg_loss = total_loss / len(test_loader)
accuracy = correct / total

print(f"\n✅ テスト損失: {avg_loss:.4f}")
print(f"✅ テスト精度: {accuracy * 100:.2f}%")

# --------------------------
# 精度ヒストグラム
# --------------------------
plt.figure(figsize=(6, 4))
plt.hist(all_preds, bins=np.arange(-0.5, 3.5, 1), rwidth=0.8, align='mid', color='skyblue')
plt.xticks([0, 1, 2], ["down", "neutral", "up"])
plt.title("Predicted Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()


# --------------------------
# 混同行列（オプション）
# --------------------------
cm = confusion_matrix(all_targets, all_preds)
# 正しい書き方：
disp = ConfusionMatrixDisplay(cm, display_labels=["up", "down", "neutral"])

disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
