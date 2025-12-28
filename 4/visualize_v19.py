import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

# 日本語フォント設定 (WindowsならMS Gothicが標準的)
plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["font.size"] = 14  # 文字サイズを大きめに



def select_file(root):
    # 初期ディレクトリを gitPython/1 にしておくと便利
    initial_dir = r"C:\Users\hrsyn\Desktop\masterPY\1"
    if not os.path.exists(initial_dir):
        initial_dir = os.getcwd()

    file_path = filedialog.askopenfilename(parent=root, initialdir=initial_dir, title="CSVファイルを選択してください", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
    return file_path



def select_channels(root):
    root.deiconify() # ウィンドウを表示
    root.title("チャンネル選択")
    
    # ウィンドウを画面中央に配置
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    w, h = 300, 350
    x = (sw - w) // 2
    y = (sh - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")

    vars = []
    checks = []
    
    # チェックボックス作成 (ch1-ch8)
    for i in range(8):
        v = tk.BooleanVar(value=False) # デフォルトでON
        vars.append(v)
        c = tk.Checkbutton(root, text=f"Channel {i+1}", variable=v, font=("MS Gothic", 12))
        c.pack(anchor="w", padx=20, pady=2)
        checks.append(c)

    selected_indices = []

    def on_ok():
        for i, v in enumerate(vars):
            if v.get():
                selected_indices.append(i)
        root.quit() # mainloopを抜ける

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=20)
    
    btn = tk.Button(btn_frame, text="OK", command=on_ok, width=10, font=("MS Gothic", 12))
    btn.pack()

    root.mainloop()

    # 何も選択されなかった場合は全部表示などの対策
    if not selected_indices:
        return list(range(8)) # 全チャンネル
    return selected_indices


def main():
    # ルートウィンドウを1つだけ作成
    root = tk.Tk()
    root.withdraw() # 最初は隠す

    csv_path = select_file(root)
    if not csv_path:
        print("ファイルが選択されませんでした。")
        root.destroy()
        return

    print(f"Loading: {csv_path}")

    # チャンネル選択ダイアログ表示
    selected_ch_indices = select_channels(root)
    print(f"Selected Channels: {[i+1 for i in selected_ch_indices]}")
    
    root.destroy() # GUI終了



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
        # 選択されたチャンネルのみループ
        for i in selected_ch_indices:
            col = f"ch{i+1}"

            if col in df.columns:
                ax.plot(df["time_rel"], df[col], label=f"EMG {col}", alpha=1.0, linewidth=1.5)

        # 振動データ (vib1_z, vib2_z)
        # 軸を分ける
        ax2 = ax.twinx()
        vib_cols = ["vib1_z", "vib2_z"]
        for col in vib_cols:
            if col in df.columns:
                ax2.plot(df["time_rel"], (df[col] - 470).abs(), label=f"Vib {col}", linestyle="-", linewidth=1.0, alpha=0.3, color="red") # 色を指定しないと自動サイクルで被る可能性があるため


        # 状態 (state) - 背景色などで表現してもいいが、今回は単純なプロットか、値として表示
        #if "state" in df.columns:
            # stateを見やすくするために少しスケールする等の工夫が可能だが、一旦そのまま
        #    ax.plot(df["time_rel"], df["state"] * 10, label="State (x10)", linestyle=":", color="black")

        ax.set_title("V19 センサーデータ可視化", fontsize=20)
        ax.set_xlabel("時間 (秒)", fontsize=16)
        ax.set_ylabel("EMG (RMS)", fontsize=16)
        ax2.set_ylabel("Vibration (Abs Diff)", fontsize=16)

        ax.tick_params(labelsize=14)
        ax2.tick_params(labelsize=14)
        ax.grid(True)
        
        # 凡例をまとめる
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc="upper right")


        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
