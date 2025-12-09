import pandas as pd
import os
import glob

# === 設定 ===
input_folder = r"C:\Users\hrsyn\Desktop\DATAforPython\emg_rms1.csv"  # 入力フォルダ
output_folder = "emg_data3/"  # 出力フォルダ
start_cut = 0.0  # 最初に削除する時間（秒）
end_cut = 29.0  # 最後に削除する時間（秒）

# 出力フォルダがなければ作成
os.makedirs(output_folder, exist_ok=True)

# CSVファイル一覧を取得
csv_files = r"C:\Users\hrsyn\Desktop\DATAforPython\emg_rms1.csv"
# csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
for file_path in [csv_files]:
    df = pd.read_csv(file_path)

    # 'Timestamp'列があるか確認
    if "Timestamp" not in df.columns:
        print(f"スキップ（Timestamp列なし）: {file_path}")
        continue

    # 最小・最大のタイムスタンプ取得（UNIXタイムスタンプ）
    min_time = df["Timestamp"].min()
    max_time = df["Timestamp"].max()

    # 切り取り範囲を計算
    start_threshold = min_time + start_cut
    end_threshold = max_time - end_cut

    # 範囲内のデータを残す
    df_trimmed = df[(df["Timestamp"] >= start_threshold) & (df["Timestamp"] <= end_threshold)].reset_index(drop=True)

    # 保存ファイル名と出力パス
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_folder, filename)

    # 保存
    df_trimmed.to_csv(output_path, index=False)
    print(f"保存完了: {output_path}")
