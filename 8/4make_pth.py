import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

data = np.load("emg_dataset_air.npz")
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print("✅ データ読み込み完了:", X_train.shape, y_train.shape)

class EMG_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3),  # (8, 20) -> (16, 18)
            nn.ReLU(),
            nn.MaxPool1d(2),                 # (16, 18) -> (16, 9)
            nn.Conv1d(16, 32, kernel_size=3), # -> (32, 7)
            nn.ReLU(),
            nn.MaxPool1d(2),                 # -> (32, 3)
            nn.Flatten(),
            nn.Linear(32 * 3, 64),           # ← 変更点！
            nn.ReLU(),
            nn.Linear(64, 3)                 # クラス数：3 (up/down/neutral)
        )

    def forward(self, x):
        return self.net(x)


# PyTorchデータローダー
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 学習ループ
model = EMG_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "cnn_emg_model_air.pth")