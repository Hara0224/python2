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
        vib_colors = ["green", "purple"]
        for i, col in enumerate(vib_cols):
            if col in df.columns:
                ax2.plot(df["time_rel"], df[col] - 470, label=f"Vib {col}", linestyle="-", linewidth=1.2, alpha=0.5, color=vib_colors[i])


        # 状態 (state) - 背景色などで表現してもいいが、今回は単純なプロットか、値として表示
        #if "state" in df.columns:
            # stateを見やすくするために少しスケールする等の工夫が可能だが、一旦そのまま
        #    ax.plot(df["time_rel"], df["state"] * 10, label="State (x10)", linestyle=":", color="black")

        ax.set_title("センサーデータ可視化", fontsize=25)
        ax.set_xlabel("時間 (秒)", fontsize=20)
        ax.set_ylabel("EMG (RMS)", fontsize=20)
        ax2.set_ylabel("Vibration", fontsize=20)

        ax.tick_params(labelsize=20)
        ax2.tick_params(labelsize=20)
        ax.grid(True)
        
        # 凡例をまとめる
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc="upper left")


        plt.tight_layout()
        
        # 0の位置を合わせる
        # 現在の表示範囲を取得
        y1_min, y1_max = ax.get_ylim()
        y2_min, y2_max = ax2.get_ylim()

        # 0を基準とした時の上下のマージン比率を計算
        # 下側(負)の広さ / 上側(正)の広さ
        # ※ データが全て正の場合、minは0付近になるため比率は小さくなる
        
        # まずは0を含むように範囲を正規化（念のため）
        y1_min = min(0, y1_min)
        y1_max = max(0, y1_max)
        y2_min = min(0, y2_min)
        y2_max = max(0, y2_max)

        # 上と下の絶対値
        top1, bot1 = y1_max, abs(y1_min)
        top2, bot2 = y2_max, abs(y2_min)

        # ゼロ除算回避
        if top1 == 0: top1 = 1.0
        if top2 == 0: top2 = 1.0

        # 比率 (bottom / top)
        r1 = bot1 / top1
        r2 = bot2 / top2

        # 大きい方の比率に合わせる
        target_r = max(r1, r2)
        
        # 新しい下限値を設定 (-top * target_r)
        new_bot1 = top1 * target_r
        new_bot2 = top2 * target_r
        
        # グラフに適用
        ax.set_ylim(-new_bot1, top1)
        ax2.set_ylim(-new_bot2, top2)

        plt.show()

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
