import pandas as pd
import os
import glob

# === 設定 ===
input_folder = r'C:\Users\hrsyn\Desktop\Python\EMG123\emg_data_11trim_abs'     # 整流化されたCSVが入っているフォルダ
output_folder = 'emg_data_11_thresh15/' # しきい値処理後のCSVを保存するフォルダ
threshold = 23                     # この値以下のデータを0にする

# 出力フォルダがなければ作成
os.makedirs(output_folder, exist_ok=True)

# フォルダ内のすべてのCSVファイルを取得
csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

# 各ファイルを処理
for file_path in csv_files:
    df = pd.read_csv(file_path)

    # しきい値処理（Timestamp列以外）
    df_thresh = df.copy()
    for col in df.columns:
        if col.lower() != 'Timestamp':
            df_thresh[col] = df_thresh[col].apply(lambda x: x if x > threshold else 0)

    # ファイル名を保ったまま保存
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_folder, filename)
    df_thresh.to_csv(output_path, index=False)

    print(f'保存完了: {output_path}')
