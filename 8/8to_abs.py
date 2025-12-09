import os
import pandas as pd

# フォルダ設定
input_folder = 'emg_data_raw'
output_folder = 'emg_data_abs'

# 出力フォルダを作成（なければ）
os.makedirs(output_folder, exist_ok=True)

# 対象の全CSVファイルを処理
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        input_path = os.path.join(input_folder, filename)
        df = pd.read_csv(input_path)

        # 絶対値化する対象列を選定（TimestampとLabel以外）
        columns_to_abs = [col for col in df.columns if col not in ['Timestamp', 'Label']]

        # データのコピーと絶対値化
        df_abs = df.copy()
        df_abs[columns_to_abs] = df_abs[columns_to_abs].abs()

        # 結果を出力
        output_path = os.path.join(output_folder, filename)
        df_abs.to_csv(output_path, index=False)
        print(f'保存完了: {output_path}')
