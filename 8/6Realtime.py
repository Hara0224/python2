from pyomyo import Myo, emg_mode
import torch
import numpy as np
import time

# =============================
# ① 学習済みCNNモデルの定義
# =============================
class EMG_CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(8, 16, kernel_size=3),   # (8, 20) → (16, 18)
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),                  # → (16, 9)
            torch.nn.Conv1d(16, 32, kernel_size=3),  # → (32, 7)
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),                  # → (32, 3)
            torch.nn.Flatten(),                     # → 32×3 = 96
            torch.nn.Linear(32 * 3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)  # ← クラス数（例: up/down/neutral）
        )

    def forward(self, x):
        return self.net(x)

# =============================
# ② モデルの読み込み
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EMG_CNN().to(device)
model.load_state_dict(torch.load("cnn_emg_model_air.pth", map_location=device))
model.eval()

# =============================
# ③ EMGデータ用バッファ
# =============================
BUFFER_SIZE = 20  # 20サンプル = 約100ms (200Hz換算)
emg_buffer = []
prev_label = None  # 直前のラベルを保持して、変化時のみ出力

# =============================
# ④ 推論関数（バッファ → ラベル）
# =============================
def predict_emg(buffer):
    x = np.array(buffer)         # shape: (20, 8)
    x = x.T                      # → shape: (8, 20)
    x = x / 128.0                # 正規化（Myoは-128〜127の整数）
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)  # shape: (1, 8, 20)
    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1).item()
    return pred

# =============================
# ⑤ EMGハンドラ（受信したとき呼ばれる）
# =============================
def emg_handler(emg, movement):
    global emg_buffer, prev_label
    if len(emg_buffer) >= BUFFER_SIZE:
        emg_buffer.pop(0)
    emg_buffer.append(emg)

    if len(emg_buffer) == BUFFER_SIZE:
        label = predict_emg(emg_buffer)

        if label != prev_label:
            prev_label = label

            if label == 0:
                label_str = "down"
            elif label == 1:
                label_str = "neutral"
            elif label == 2:
                label_str = "up"
            else:
                label_str = "unknown"

            print(f"判定: {label_str}")

# =============================
# ⑥ Myo接続・ループ開始
# =============================
m = Myo(mode=emg_mode.RAW)
m.connect()
m.add_emg_handler(emg_handler)
print("開始：Ctrl+Cで停止")

try:
    while True:
        m.run()
        time.sleep(0.01)
except KeyboardInterrupt:
    print("終了しました。")
