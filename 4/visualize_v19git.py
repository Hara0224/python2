import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

# 日本語フォント設定 (WindowsならMS Gothicが標準的)
plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["font.size"] = 20  # 文字サイズを大きめに


def select_file(root):
    # 初期ディレクトリを gitPython/1 にしておくと便利
    initial_dir = r"C:\Users\hrsyn\Desktop\masterPY\1"
    if not os.path.exists(initial_dir):
        initial_dir = os.getcwd()

    file_path = filedialog.askopenfilename(parent=root, initialdir=initial_dir, title="CSVファイルを選択してください", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
    return file_path


def select_channels(root):
    root.deiconify()  # ウィンドウを表示
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
        v = tk.BooleanVar(value=False)  # デフォルトでON
        vars.append(v)
        c = tk.Checkbutton(root, text=f"Channel {i+1}", variable=v, font=("MS Gothic", 12))
        c.pack(anchor="w", padx=20, pady=2)
        checks.append(c)

    selected_indices = []

    def on_ok():
        for i, v in enumerate(vars):
            if v.get():
                selected_indices.append(i)
        root.quit()  # mainloopを抜ける

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=20)

    btn = tk.Button(btn_frame, text="OK", command=on_ok, width=10, font=("MS Gothic", 12))
    btn.pack()

    root.mainloop()

    # 何も選択されなかった場合は全部表示などの対策
    if not selected_indices:
        return list(range(8))  # 全チャンネル
    return selected_indices


def main():
    # ルートウィンドウを1つだけ作成
    root = tk.Tk()
    root.withdraw()  # 最初は隠す

    csv_path = select_file(root)
    if not csv_path:
        print("ファイルが選択されませんでした。")
        root.destroy()
        return

    print(f"Loading: {csv_path}")

    # チャンネル選択ダイアログ表示
    selected_ch_indices = select_channels(root)
    print(f"Selected Channels: {[i+1 for i in selected_ch_indices]}")

    root.destroy()  # GUI終了

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

        # ===== 追加: キャリブレーション値の計算 (V19 Z-scoreロジック) =====
        import numpy as np

        # V19のパラメータ (Z-score版)

        DOWN_CH_IDX = [5, 6]  # ch6, ch7
        # UP_HOLD_SENSITIVITY = 5.0 # 不使用
        STRONG_RATIO = 10.0
        CALIB_DURATION = 3.0
        K_SIGMA = 5.7  # Trigger Threshold (Sigma)

        print("\n=== Calculating Thresholds from first 5.0s ===")
        # 最初の5秒間のデータを抽出
        calib_df = df[df["time_rel"] <= CALIB_DURATION]

        # 可視化用の閾値 (Raw値に換算)
        trigger_thresholds = np.zeros(8)
        strong_threshold = 0.0

        if not calib_df.empty:
            cal_means = np.zeros(8)
            cal_stds = np.zeros(8)

            # 各チャンネルのMean, Stdを計算
            for i in range(8):
                col_name = f"ch{i+1}"
                if col_name in calib_df.columns:
                    val = calib_df[col_name].values
                    # 平均・標準偏差
                    m = np.mean(val)
                    s = np.std(val, ddof=1)
                    if s < 1e-6:
                        s = 1e-6

                    cal_means[i] = m
                    cal_stds[i] = s

                    # Trigger Threshold (Raw scale) = Mean + K_SIGMA * Std
                    trigger_thresholds[i] = m + s * K_SIGMA

            # Strong Threshold (DOWN_CHの平均Mean + DOWN_CHの標準偏差*Ratio の平均)
            # V19: strong_thr_list.append(mean[ch] + std[ch] * STRONG_RATIO)
            # strong_threshold = np.mean(strong_thr_list)
            strong_vals = []
            for idx in DOWN_CH_IDX:
                val = cal_means[idx] + cal_stds[idx] * STRONG_RATIO
                strong_vals.append(val)

            if strong_vals:
                strong_threshold = np.mean(strong_vals)

            print(f"Trigger Thresholds (Mean + {K_SIGMA}sigma): {trigger_thresholds}")
            print(f"Strong Threshold (Mean + {STRONG_RATIO}sigma): {strong_threshold:.2f}")
        else:
            print("Warning: Not enough data for calibration calculation.")

        # グラフ作成
        fig, ax = plt.subplots(figsize=(12, 8))

        # EMGデータ (ch1-ch8)
        # 選択されたチャンネルのみループ
        for i in selected_ch_indices:
            col = f"ch{i+1}"

            if col in df.columns:
                # プロット時にLineオブジェクトを取得して色を合わせる
                lines = ax.plot(df["time_rel"], df[col], label=f"EMG {col}", alpha=1.0, linewidth=1.5)
                line_color = lines[0].get_color()

                # ★追加: 閾値の描画
                # Trigger Threshold (K_SIGMA)
                t_thr = trigger_thresholds[i]
                if t_thr > 0:
                    ax.axhline(y=t_thr, color=line_color, linestyle="--", linewidth=1.0, alpha=0.7, label=f"{col} Trig ({t_thr:.1f})")

        # Strong Threshold (共通) の描画
        # もし選択されたチャンネルにDOWN_CHが含まれていれば表示するなど制御してもいいが、
        # とりあえず赤線で引いておく
        is_down_ch_selected = any(idx in DOWN_CH_IDX for idx in selected_ch_indices)
        if is_down_ch_selected:
            ax.axhline(y=strong_threshold, color="red", linestyle="--", linewidth=1.5, label=f"Strong Thr ({strong_threshold:.1f})")

        # 振動データ (vib1_z, vib2_z)
        # 軸を分ける
        ax2 = ax.twinx()
        vib_cols = ["vib1_z", "vib2_z"]
        vib_colors = ["red", "blue"]

        for i, col in enumerate(vib_cols):
            if col in df.columns:
                # 振動データ (生データ - 470)
                v_data = df[col] - 470
                ax2.plot(df["time_rel"], v_data, label=f"Vib {col}", linestyle="-", linewidth=1.2, alpha=0.5, color=vib_colors[i])

        # ===== Latency Analysis (Trigger -> Max Vib) =====
        # 1. Trigger地点の探索 (Calibration後)
        # DOWN_CH_IDX (5, 6) のいずれかが Trigger Threshold を超えた最初の点

        # 検証範囲: キャリブレーション終了後
        analysis_df = df[df["time_rel"] > CALIB_DURATION].copy()

        first_trigger_time = None
        trigger_ch_name = ""

        if not analysis_df.empty and not calib_df.empty:
            # Z-scoreを計算して判定
            for idx in range(len(analysis_df)):
                row_idx = analysis_df.index[idx]
                t = analysis_df["time_rel"].loc[row_idx]

                is_triggered = False
                for ch_idx in DOWN_CH_IDX:
                    # ch_idx は 0-indexed, columnsは ch1..ch8
                    col = f"ch{ch_idx+1}"
                    if col in df.columns:
                        val = df[col].loc[row_idx]
                        mean_val = cal_means[ch_idx]
                        std_val = cal_stds[ch_idx]

                        # Z-score Check
                        if (val - mean_val) / std_val > K_SIGMA:
                            first_trigger_time = t
                            trigger_ch_name = col
                            is_triggered = True
                            break
                if is_triggered:
                    break

        # 2. Vibration Peakの探索 (自動検出は削除)
        # ユーザー要望によりピークの自動検出とレイテンシ表示は削除
        # Trigger Point の線だけ残す

        if first_trigger_time is not None:
            # 3. 可視化
            # Trigger Line (Vertical)
            ax.axvline(x=first_trigger_time, color="orange", linestyle="-.", linewidth=1.5, label="Trigger Point")
            print(f"Detected Trigger at {first_trigger_time:.3f}s ({trigger_ch_name})")

        # 状態 (state) - 背景色などで表現してもいいが、今回は単純なプロットか、値として表示
        # if "state" in df.columns:
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
        # 重複を除く (閾値線などでラベルが増えすぎるのを防ぐため、必要ならsetでユニークにするが、今回はそのまま)
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc="upper right")

        # ===== Interactive Manual Measurement =====
        click_state = {"count": 0, "start_time": None, "start_val": None, "markers": []}

        def onclick(event):
            # プロットエリア外のクリックは無視
            if event.inaxes not in [ax, ax2]:
                return

            # 左クリック (1) のみ反応
            if event.button != 1:
                return

            t = event.xdata
            y = event.ydata

            # カウンターを進める
            click_state["count"] += 1
            count = click_state["count"]

            if count == 1:
                # 1点目: Start
                click_state["start_time"] = t
                click_state["start_val"] = y

                # 古いマーカーがあれば消す (3回目クリック時のリセット相当を兼ねる)
                for m in click_state["markers"]:
                    try:
                        m.remove()
                    except:
                        pass
                click_state["markers"] = []

                # 緑の点プロット
                # どちらの軸をクリックしたかによってy座標の扱いが変わるが、x軸(時間)が重要
                (marker,) = ax.plot(t, y, "go", markersize=8, zorder=10, transform=event.inaxes.transData)  # clickした軸上の座標
                click_state["markers"].append(marker)
                print(f"[Manual] Start Point: {t:.3f} s")
                plt.draw()

            elif count == 2:
                # 2点目: End
                start_t = click_state["start_time"]

                # 赤の点は削除
                # marker, = ax.plot(t, y, 'ro', markersize=8, zorder=10, transform=event.inaxes.transData)
                # click_state["markers"].append(marker)

                # 差分計算
                diff_ms = abs(t - start_t) * 1000
                print(f"[Manual] End Point:   {t:.3f} s")
                print(f"[Manual] Diff:        {diff_ms:.1f} ms")

                # 矢印プロット (start -> end)
                # 異なる軸間だとY座標が合わない問題があるが、annotateのxy, xytextでなんとかする
                # 一旦、クリックされた軸座標系を使う
                ann = event.inaxes.annotate(
                    f"Manual: {diff_ms:.1f}ms",
                    xy=(t, y),
                    xytext=(start_t, click_state["start_val"]),
                    arrowprops=dict(arrowstyle="<->", color="blue", linewidth=2),
                    fontsize=12,
                    color="blue",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8),
                )
                click_state["markers"].append(ann)
                plt.draw()

            else:
                # 3回目: リセットして1点目として扱う
                click_state["count"] = 1

                # 全削除
                for m in click_state["markers"]:
                    try:
                        m.remove()
                    except:
                        pass
                click_state["markers"] = []

                # Startとして記録
                click_state["start_time"] = t
                click_state["start_val"] = y
                (marker,) = ax.plot(t, y, "go", markersize=8, zorder=10, transform=event.inaxes.transData)
                click_state["markers"].append(marker)
                print(f"[Manual] Start Point (Reset): {t:.3f} s")
                plt.draw()

        fig.canvas.mpl_connect("button_press_event", onclick)

        plt.tight_layout()

        # 0の位置を合わせる
        # 現在の表示範囲を取得
        y1_min, y1_max = ax.get_ylim()
        y2_min, y2_max = ax2.get_ylim()

        # 0を基準とした時の上下のマージン比率を計算
        # 下側(負)の広さ / 上側(正)の広さ
        
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
