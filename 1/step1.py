from pyomyo import Myo, emg_mode
import time
import csv
import os
from datetime import datetime
import multiprocessing  # データの取得と処理を分離
import queue  # プロセス間でデータを安全に受け渡す

# === 設定 ===
labels = ["radial_dev", "ulnar_dev", "rest"]
repeats_per_label = 6
record_duration = 2.0  # 1回あたりの記録時間（秒）
interval_between = 1.0  # 各収録間の休憩（秒）
save_dir = "./emg_data_multiprocess/"  # 保存先ディレクトリを更新

# 保存先フォルダがなければ作成
os.makedirs(save_dir, exist_ok=True)

# === プロセス間通信キューのセットアップ ===
# MyoプロセスからメインプロセスへEMGデータを送るためのキュー
data_queue = multiprocessing.Queue()


# === Myo Worker プロセス関数 ===
def myo_worker(q):
    """Myoデバイスとの接続、データ取得、キューへの投入を担うプロセス"""
    print("[INFO] Myo Workerプロセス開始...")

    # Myo初期化。動作実績のある RAW モードを使用
    m = Myo(mode=emg_mode.FILTERED)

    try:
        # 接続を試みる
        m.connect()

        # 接続確認 (成功すれば続行)
        m.set_leds([128, 0, 0], [0, 0, 0])
        m.vibrate(1)
        print("[INFO] Myo Worker: 接続成功。データストリーム開始。")

    except Exception as e:
        print(f"❌ Myo Worker: 接続失敗。プロセスを終了します: {e}")
        return  # 接続失敗時はプロセスを終了

    def add_to_queue(emg, movement):
        """Myoからデータを受信するたびにキューに投入するハンドラ"""
        timestamp = time.time()
        # [タイムスタンプ, CH1, ..., CH8] の形式でキューに格納
        q.put([timestamp] + list(emg))

    m.add_emg_handler(add_to_queue)

    # データストリームを継続的に実行
    while True:
        try:
            m.run()
        except KeyboardInterrupt:
            print("[INFO] Myo Worker: ユーザー操作により終了。")
            break
        except Exception as e:
            # Myoの接続が切れた場合などのエラー処理
            print(f"[ERROR] Myo Worker: ランタイムエラー発生: {e}")
            time.sleep(1.0)
            continue

    # プロセス終了時にMyoを切断
    try:
        m.disconnect()
    except AttributeError:
        pass  # m.disconnect() がないバージョンに対応
    print("[INFO] Myo Workerプロセス終了。")


# === メイン収録ループ ===
if __name__ == "__main__":

    # Myo Workerプロセスを開始
    myo_process = multiprocessing.Process(target=myo_worker, args=(data_queue,))
    myo_process.start()

    # Workerが接続を完了するまで待機（ポーリング）
    print("Myo接続待機中...")
    while data_queue.empty():
        # キューにデータが入るまで（＝Myoがデータ送信を開始するまで）待つ
        time.sleep(0.1)
    print("Myoデータ受信確認。収録を開始します。")

    print(
        f"\n🟢 記録開始準備OK（{record_duration}秒記録 / {interval_between}秒休憩 × 各ラベル{repeats_per_label}回）"
    )

    counter = 1
    total = len(labels) * repeats_per_label

    for i in range(repeats_per_label):
        for label in labels:

            print(
                f"\n🔴 [{counter}/{total}] {label} を記録します...（{record_duration}秒間動作）"
            )

            # 収録前にキューを空にして、古いデータを破棄
            raw_data = []
            while not data_queue.empty():
                try:
                    data_queue.get_nowait()
                except queue.Empty:
                    break

            start_time = time.time()

            # 指定された記録時間が経過するまで、キューからデータを取得し続ける
            while time.time() < start_time + record_duration:
                try:
                    # ノンブロッキングでキューからデータを取得
                    emg_data = data_queue.get_nowait()
                    raw_data.append(emg_data)
                except queue.Empty:
                    # キューが空の場合、わずかに待機
                    time.sleep(0.001)

            # === 保存 ===
            timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{save_dir}emgraw_{label}_{timestamp_str}.csv"

            if len(raw_data) > 0:
                with open(filename, mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        ["Timestamp"] + [f"CH{j+1}" for j in range(8)] + ["Label"]
                    )
                    for row in raw_data:
                        # rowは [timestamp, ch1...ch8] なので、[label]を追加
                        writer.writerow(row + [label])
                print(f"✅ 保存完了: {filename}（{len(raw_data)}サンプル）")
            else:
                print(
                    f"⚠️ 警告: {label} の記録で0サンプルを検出しました。このファイルはスキップされました。"
                )

            counter += 1
            print(f"⏸️ 休憩中です...（{interval_between}秒）")
            time.sleep(interval_between)

    # === 終了処理 ===
    print("\n✅ すべての収録が完了しました！")

    # Myo Workerプロセスを終了させる
    if myo_process.is_alive():
        myo_process.terminate()
        myo_process.join()

    print("プログラム終了。")
