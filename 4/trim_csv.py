import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
import sys


def select_file(root):

    # ファイル選択ダイアログを表示し、ユーザーにCSVファイルを選択させる。
    # 初期ディレクトリは C:\Users\hrsyn\Desktop\gitPython\1 またはカレントディレクトリ。
    initial_dir = r"C:\Users\hrsyn\Desktop\masterPY\1"
    if not os.path.exists(initial_dir):
        initial_dir = os.getcwd()

    file_path = filedialog.askopenfilename(parent=root, initialdir=initial_dir, title="処理するCSVファイルを選択してください", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
    return file_path


def main():
    # GUI用のルートウィンドウ作成（表示はしない）
    root = tk.Tk()
    root.withdraw()

    print("=== EMGデータ トリミングツール ===")

    # 1. ファイル選択
    input_path = select_file(root)
    if not input_path:
        print("ファイルが選択されませんでした。終了します。")
        root.destroy()
        return

    print(f"選択されたファイル: {input_path}")
    root.destroy()  # ファイル選択が終わったらGUI破棄

    # 2. トリミング時間の入力
    try:
        trim_start_sec = float(input("開始から削除する時間 (秒) を入力してください [例: 5]: "))
        trim_end_sec = float(input("終了から削除する時間 (秒) を入力してください [例: 5]: "))
    except ValueError:
        print("エラー: 数値を入力してください。")
        return

    try:
        # 3. データの読み込み
        print("ファイルを読み込み中...")
        df = pd.read_csv(input_path)

        if "timestamp" not in df.columns:
            print("エラー: CSVファイルに 'timestamp' 列が見つかりません。")
            return

        # 4. 相対時間の計算
        start_timestamp = df["timestamp"].iloc[0]
        end_timestamp = df["timestamp"].iloc[-1]

        # trim_start_sec 分だけ進んだ時刻
        trim_start_abs = start_timestamp + trim_start_sec
        # trim_end_sec 分だけ戻った時刻
        trim_end_abs = end_timestamp - trim_end_sec

        total_duration = end_timestamp - start_timestamp
        print(f"データ長: {total_duration:.2f}秒")
        print(f"トリミング範囲: {trim_start_sec}秒 ～ (末尾 -{trim_end_sec}秒)")

        if trim_start_abs >= trim_end_abs:
            print("エラー: トリミング後のデータが残りません。指定時間を短くしてください。")
            return

        # 5. フィルタリング (timestamp列で判定)
        # trim_start_abs より大きく、trim_end_abs より小さいデータを残す
        df_trimmed = df[(df["timestamp"] >= trim_start_abs) & (df["timestamp"] <= trim_end_abs)]

        if df_trimmed.empty:
            print("警告: 条件に合うデータがありませんでした。")
            return

        # 6. 保存
        input_dir = os.path.dirname(input_path)
        input_filename = os.path.basename(input_path)
        filename_no_ext = os.path.splitext(input_filename)[0]
        output_filename = f"{filename_no_ext}_trimmed.csv"
        output_path = os.path.join(input_dir, output_filename)

        df_trimmed.to_csv(output_path, index=False)
        print(f"\n保存完了: {output_path}")
        print(f"行数: {len(df)} -> {len(df_trimmed)}")

    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()

    input("\nEnterキーを押して終了...")


if __name__ == "__main__":
    main()
