from pyomyo import Myo, emg_mode
import time
import csv
import os
from datetime import datetime

# === 設定 ===
# 動作ラベルを撓屈/尺屈/静止に変更 (以前の議論と整合)
labels = ["radial_dev", "ulnar_dev", "rest"]
repeats_per_label = 6  # 1ラベルあたりの繰り返し回数 (例: 6回)
record_duration = 5.0  # 1回あたりの記録時間（秒）
interval_between = 3.0  # 各収録間の休憩（秒）
save_dir = "./emg_data_raw/"  # 保存先ディレクトリ

# 保存先フォルダがなければ作成
os.makedirs(save_dir, exist_ok=True)

# === Myo初期化 ===
m = Myo(mode=emg_mode.RAW)
m.connect()
if not m.is_connected():
    print("❌ Myoデバイスが見つからないか、接続に失敗しました。プログラムを終了します。")
    exit()

m.set_leds([128, 0, 0], [0, 0, 0])
m.vibrate(1)
print("[INFO] Myo接続完了。RAWモードでストリーミング開始。")

# === EMGデータ収集用グローバル変数 ===
raw_data = []  # 記録対象のデータを一時的に保持
current_label_name = ""  # 現在記録中のラベル名
is_recording = False  # 記録フラグ


# === EMGハンドラ ===
def collect_emg(emg, movement):
    """MyoからEMGデータを受信するたびに呼び出されるハンドラ"""
    global raw_data, is_recording

    if is_recording:
        timestamp = time.time()
        # EMGデータ（8要素）をリストに追加
        raw_data.append([timestamp] + list(emg))


# ハンドラを設定
m.add_emg_handler(collect_emg)

# バックグラウンドでEMGストリームのイベント処理を開始
m.run_in_background(True)

# === 記録ループ ===
print(f"\n🟢 記録開始準備OK（{record_duration}秒記録 / {interval_between}秒休憩 × 各ラベル{repeats_per_label}回）")

counter = 1
total = len(labels) * repeats_per_label

for i in range(repeats_per_label):
    for label in labels:
        current_label_name = label  # 現在のラベル名を設定

        print(f"\n🔴 [{counter}/{total}] {label} を記録します...（{record_duration}秒間動作）")
        m.vibrate(2)
        raw_data = []  # 直前のデータをクリア

        # 記録開始
        is_recording = True
        start_time = time.time()

        # 指定された記録時間が経過するまで待機
        time.sleep(record_duration)

        # 記録終了
        is_recording = False

        # === 保存 ===
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{save_dir}emgraw_{label}_{timestamp}.csv"

        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp"] + [f"CH{j+1}" for j in range(8)] + ["Label"])
            for row in raw_data:
                # rowは [timestamp, ch1...ch8] なので、[label]を追加
                writer.writerow(row + [label])

        print(f"✅ 保存完了: {filename}（{len(raw_data)}サンプル）")

        counter += 1
        print(f"⏸️ 休憩中です...（{interval_between}秒）")
        time.sleep(interval_between)


m.run_in_background(False)  # バックグラウンド処理を停止
m.vibrate(3)
print("\n✅ すべての収録が完了しました！")
