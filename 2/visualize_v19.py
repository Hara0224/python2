import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

# 日本語フォント設定 (WindowsならMS Gothicが標準的)
plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["font.size"] = 14  # 文字サイズを大きめに


def select_file():
    root = tk.Tk()
    root.withdraw()
    # 初期ディレクトリを gitPython/1 にしておくと便利
    initial_dir = r"c:\Users\hrsyn\Desktop\gitPython\1"
    if not os.path.exists(initial_dir):
        initial_dir = os.getcwd()

    file_path = filedialog.askopenfilename(initialdir=initial_dir, title="CSVファイルを選択してください", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
    return file_path


def main():
    csv_path = select_file()
    if not csv_path:
        print("ファイルが選択されませんでした。")
        return

    print(f"Loading: {csv_path}")

    try:
        # CSV読み込み
        df = pd.read_csv(csv_path)

        # タイムスタンプを相対時間に変換 (開始0秒)
        if "timestamp" in df.columns:
            start_time = df["timestamp"].iloc[0]
            df["time_rel"] = df["timestamp"] - start_time
        else:
            # timestampがない場合は行番号を時間とする
            df["time_rel"] = df.index * 0.05  # 仮のサンプリングレート

        # グラフ作成
        fig, ax = plt.subplots(figsize=(12, 8))

        # EMGデータ (ch1-ch8)
        emg_cols = [f"ch{i+1}" for i in range(8)]
        for col in emg_cols:
            if col in df.columns:
                ax.plot(df["time_rel"], df[col], label=f"EMG {col}", alpha=0.7, linewidth=1.5)

        # 振動データ (vib1_z, vib2_z)
        # 軸を分けるか悩みますが、とりあえず「重ねて」指示通り同じ軸で描画します。
        # スケールが違いすぎる場合は調整が必要ですが、一旦そのまま
        vib_cols = ["vib1_z", "vib2_z"]
        for col in vib_cols:
            if col in df.columns:
                ax.plot(df["time_rel"], df[col], label=f"Vib {col}", linestyle="--", linewidth=2.0)

        # 状態 (state) - 背景色などで表現してもいいが、今回は単純なプロットか、値として表示
        if "state" in df.columns:
            # stateを見やすくするために少しスケールする等の工夫が可能だが、一旦そのまま
            ax.plot(df["time_rel"], df["state"] * 10, label="State (x10)", linestyle=":", color="black")

        ax.set_title("V19 センサーデータ可視化", fontsize=20)
        ax.set_xlabel("時間 (秒)", fontsize=16)
        ax.set_ylabel("センサー値 / 状態", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.grid(True)
        ax.legend(fontsize=12, loc="upper right")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
