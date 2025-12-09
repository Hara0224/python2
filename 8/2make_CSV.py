from pyomyo import Myo, emg_mode
import time
import csv
import os
from datetime import datetime

# === 設定 ===
labels = ["updown"]
repeats_per_label = 1  #
record_duration = 2  # 1回あたりの記録時間（秒）
interval_between = 1.0  # 各収録間の休憩（秒）
save_dir = "./emg_data_raw_1/"  # 保存先ディレクトリ

# 保存先フォルダがなければ作成
os.makedirs(save_dir, exist_ok=True)

# === Myo初期化 ===
m = Myo(mode=emg_mode.RAW)
m.connect()
m.set_leds([128, 0, 0], [0, 0, 0])
m.vibrate(1)

# === EMGハンドラ ===
raw_data = []


def collect_emg(emg, movement):
    timestamp = time.time()
    if isinstance(emg, (list, tuple)) and len(emg) == 8:
        raw_data.append([timestamp] + list(emg))


m.add_emg_handler(collect_emg)

# === 記録ループ ===
print(" 記録開始準備")

counter = 1
total = len(labels) * repeats_per_label

for i in range(repeats_per_label):
    for label in labels:
        print(
            f"\n🔴 [{counter}/{total}] {label} を記録します...（{record_duration}秒）"
        )
        m.vibrate(2)
        raw_data = []  # 直前のデータをクリア

        # 記録開始
        start_time = time.time()
        while time.time() - start_time < record_duration:
            m.run()

        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}emg_raw_{label}_{timestamp}.csv"
        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp"] + [f"CH{i+1}" for i in range(8)] + ["Label"])
            for row in raw_data:
                writer.writerow(row + [label])

        print(f"✅ 保存完了: {filename}（{len(raw_data)}サンプル）")
        counter += 1
        time.sleep(interval_between)

m.vibrate(3)
print("\n完了")
