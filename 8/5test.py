import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

# === EMG_CNN モデル定義 ===
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
            nn.Linear(32 * 11, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# === データ読み込み ===
data = np.load("emg_dataset.npz")
X_test = torch.tensor(data["X_test"], dtype=torch.float32)
y_test = torch.tensor(data["y_test"], dtype=torch.long)

# === データローダー作成 ===
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32)

# === モデルの読み込み・初期化 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EMG_CNN().to(device)

# （学習済みモデルがある場合は以下を使う）
# model.load_state_dict(torch.load("cnn_emg_model.pth"))

# ここではランダム初期化（精度は参考値）
model.eval()

# === 推論・精度評価 ===
y_true = []
y_pred = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb)
        pred_labels = preds.argmax(dim=1).cpu().numpy()
        y_true.extend(yb.numpy())
        y_pred.extend(pred_labels)

# === 精度と混同行列 ===
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print(f"✅ テスト精度: {acc * 100:.2f}%")
print("混同行列:\n", cm)
