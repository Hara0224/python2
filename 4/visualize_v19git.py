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
    w, h = 300, 450 # 高さ拡張
    x = (sw - w) // 2
    y = (sh - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")

    vars_emg = []
    checks_emg = []

    # EMG Label
    lbl_emg = tk.Label(root, text="EMG Channels", font=("MS Gothic", 10, "bold"))
    lbl_emg.pack(anchor="w", padx=20, pady=(10, 0))

    # チェックボックス作成 (ch1-ch8)
    for i in range(8):
        v = tk.BooleanVar(value=False)  # デフォルトでOFF
        vars_emg.append(v)
        c = tk.Checkbutton(root, text=f"Channel {i+1}", variable=v, font=("MS Gothic", 12))
        c.pack(anchor="w", padx=20, pady=2)
        checks_emg.append(c)

    # Vib Label
    lbl_vib = tk.Label(root, text="Vibration Sensors", font=("MS Gothic", 10, "bold"))
    lbl_vib.pack(anchor="w", padx=20, pady=(10, 0))

    vars_vib = []
    vib_labels = ["vib1_z", "vib2_z"]
    for i, label in enumerate(vib_labels):
        v = tk.BooleanVar(value=True) # デフォルトでON
        vars_vib.append(v)
        c = tk.Checkbutton(root, text=label, variable=v, font=("MS Gothic", 12))
        c.pack(anchor="w", padx=20, pady=2)

    selected_emg_indices = []
    selected_vib_indices = []

    def on_ok():
        for i, v in enumerate(vars_emg):
            if v.get():
                selected_emg_indices.append(i)
        for i, v in enumerate(vars_vib):
             if v.get():
                 selected_vib_indices.append(i)
        root.quit()  # mainloopを抜ける

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=20)

    btn = tk.Button(btn_frame, text="OK", command=on_ok, width=10, font=("MS Gothic", 12))
    btn.pack()

    root.mainloop()

    # 何も選択されなかった場合は全部表示などの対策 (EMGのみ)
    if not selected_emg_indices:
        selected_emg_indices = list(range(8))  # 全チャンネル
    
    # 振動は選択なければ空リストでOK
    return selected_emg_indices, selected_vib_indices


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
    # チャンネル選択ダイアログ表示
    selected_ch_indices, selected_vib_indices = select_channels(root)
    print(f"Selected EMG Channels: {[i+1 for i in selected_ch_indices]}")
    print(f"Selected Vib Channels: {selected_vib_indices}")

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

        DOWN_CH_IDX = [5]  # ch6, ch7
        # UP_HOLD_SENSITIVITY = 5.0 # 不使用
        STRONG_RATIO = 5.0
        CALIB_DURATION = 5.0
        K_SIGMA = 5.9  # Trigger Threshold (Sigma)
        MEASURE_DURATION_MS = 140 # Measurement window for Strong/Weak check
        COOLDOWN_MS = 500         # Cooldown period

        # New Calibration Timeline
        # 0-5s: Resting
        # 5-8s: Wait
        # 8-11s: Weak
        # 11-14s: Wait
        # 14-17s: Strong
        # 17s+: Analysis
        
        calib_rest_end = 5.0
        calib_weak_start = 8.0
        calib_weak_end = 11.0
        calib_strong_start = 14.0
        calib_strong_end = 17.0

        print("\n=== Calculating Thresholds ===")
        
        # 1. Resting Calibration (0-5s)
        rest_df = df[df["time_rel"] <= calib_rest_end]
        
        cal_means = np.zeros(8)
        cal_stds = np.zeros(8)
        
        if not rest_df.empty:
            for i in range(8):
                col_name = f"ch{i+1}"
                if col_name in rest_df.columns:
                    val = rest_df[col_name].values
                    cal_means[i] = np.mean(val)
                    cal_stds[i] = np.std(val, ddof=1)
                    if cal_stds[i] < 1e-6: cal_stds[i] = 1e-6
        else:
             print("Warning: No Resting Data found.")

        # Trigger Thresholds (Base)
        trigger_thresholds = cal_means + cal_stds * K_SIGMA
        print(f"Trigger Thresholds (Mean + {K_SIGMA}sigma): {trigger_thresholds}")

        # 2. Weak Calibration (8-11s)
        weak_df = df[(df["time_rel"] >= calib_weak_start) & (df["time_rel"] <= calib_weak_end)]
        weak_max = 0.0
        if not weak_df.empty:
             # DOWN_CH_IDX = [5, 6] (ch6, ch7)
             vals = []
             for idx in DOWN_CH_IDX:
                 col = f"ch{idx+1}"
                 if col in weak_df.columns:
                     vals.append(weak_df[col].max())
             if vals:
                 weak_max = max(vals)
        print(f"Weak Max (8-11s): {weak_max:.2f}")

        # 3. Strong Calibration (14-17s)
        strong_df = df[(df["time_rel"] >= calib_strong_start) & (df["time_rel"] <= calib_strong_end)]
        strong_max = 0.0
        if not strong_df.empty:
             vals = []
             for idx in DOWN_CH_IDX:
                 col = f"ch{idx+1}"
                 if col in strong_df.columns:
                     vals.append(strong_df[col].max())
             if vals:
                 strong_max = max(vals)
        print(f"Strong Max (14-17s): {strong_max:.2f}")

        # 4. Calculate Strong Threshold
        strong_threshold = 0.0
        if strong_max > 0:
            strong_threshold = weak_max + (strong_max - weak_max) * 0.5
        else:
            # Fallback if no strong data (e.g. short recording)
             vals = []
             for idx in DOWN_CH_IDX:
                vals.append(cal_means[idx] + cal_stds[idx] * STRONG_RATIO)
             if vals:
                 strong_threshold = np.mean(vals)
             print("Warning: Using default Strong Threshold (no strong calib data).")

        print(f"Strong Threshold: {strong_threshold:.2f}")

        # グラフ作成
        fig, ax = plt.subplots(figsize=(12, 8))

        # 表示用マスク (17秒以降のみ表示)
        plot_mask = df["time_rel"] > 17.0

        # EMGデータ (ch1-ch8)
        # 選択されたチャンネルのみループ
        # 色のリスト (緑, 紫)
        colors = ["green", "purple"]

        for idx, i in enumerate(selected_ch_indices):
            col = f"ch{i+1}"
            plot_color = colors[idx % len(colors)]

            if col in df.columns:
                # プロット時にLineオブジェクトを取得して色を合わせる
                # 17秒以降のみプロット
                lines = ax.plot(df.loc[plot_mask, "time_rel"], df.loc[plot_mask, col], label=f"筋電 {col}", alpha=1.0, linewidth=1.5, color=plot_color)
                line_color = lines[0].get_color()

                # ★追加: 閾値の描画
                # Trigger Threshold (K_SIGMA)
                t_thr = trigger_thresholds[i]
                if t_thr > 0:
                    ax.axhline(y=t_thr, color=line_color, linestyle="--", linewidth=1.0, alpha=0.7, label=f"{col} 閾値 ({t_thr:.1f})")

        # Strong Threshold (共通) の描画
        # もし選択されたチャンネルにDOWN_CHが含まれていれば表示するなど制御してもいいが、
        # とりあえず赤線で引いておく
        is_down_ch_selected = any(idx in DOWN_CH_IDX for idx in selected_ch_indices)
        if is_down_ch_selected:
            ax.axhline(y=strong_threshold, color="red", linestyle="--", linewidth=1.5, label=f"強閾値 ({strong_threshold:.1f})")

        # 振動データ (vib1_z, vib2_z)
        # 軸を分ける
        ax2 = ax.twinx()
        vib_cols = ["vib1_z", "vib2_z"]
        vib_colors = ["red", "blue"]

        for i, col in enumerate(vib_cols):
            # iが選択されたindexリストに含まれているか確認
            if i in selected_vib_indices and col in df.columns:
                # 振動データ (生データ - 470)
                v_data = df.loc[plot_mask, col] - 470
                ax2.plot(df.loc[plot_mask, "time_rel"], v_data, label=f"振動 {col}", linestyle="-", linewidth=1.2, alpha=0.5, color=vib_colors[i])

        # ===== Latency Analysis (Trigger -> Max Vib) =====
        # 1. Trigger地点の探索 (Calibration後)
        # DOWN_CH_IDX (5, 6) のいずれかが Trigger Threshold を超えた最初の点

        # 検証範囲: キャリブレーション終了後
        analysis_df = df[df["time_rel"] > CALIB_DURATION].copy()

        # Calibration Windows Visualization
        # Weak (Blue), Strong (Red)
        # 17秒以降のみ表示するため、キャリブレーション範囲の表示はコメントアウト
        # ax.axvspan(calib_weak_start, calib_weak_end, color="blue", alpha=0.1, label="キャリブ: 弱")
        # ax.axvspan(calib_strong_start, calib_strong_end, color="red", alpha=0.1, label="キャリブ: 強")

        # 1. Trigger地点の探索 (Calibration後 = 17s以降)
        # DOWN_CH_IDX (5, 6) のいずれかが Trigger Threshold を超えた最初の点
        # V19 logic: Attack(Measure 200ms) -> Cooldown(500ms) -> Idle

        # 検証範囲: Strong Calibration終了後
        analysis_df = df[df["time_rel"] > calib_strong_end].copy()

        trigger_events = [] # List of (time, ch_name)
        next_available_time = 0.0

        if not analysis_df.empty and not rest_df.empty:
            # Z-scoreを計算して判定
            for idx in range(len(analysis_df)):
                row_idx = analysis_df.index[idx]
                t = analysis_df["time_rel"].loc[row_idx]

                if t < next_available_time:
                    continue

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
                            trigger_events.append((t, col))
                            # Update next available time
                            next_available_time = t + (MEASURE_DURATION_MS + COOLDOWN_MS) / 1000.0
                            is_triggered = True
                            break
    
        # 2. Vibration Peakの探索 (自動検出は削除)
        # ユーザー要望によりピークの自動検出とレイテンシ表示は削除
        # Trigger Point の線だけ残す

        if trigger_events:
            print(f"Detected {len(trigger_events)} triggers.")
            # 3. 可視化
            for i, (trig_time, trig_ch) in enumerate(trigger_events):
                # Trigger Line (Vertical)
                label_name = "トリガー" if i == 0 else None # Legendには1回だけ表示
                ax.axvline(x=trig_time, color="orange", linestyle="-.", linewidth=1.5, label=label_name)
                
                # Measurement Duration (Shaded Area)
                measure_end_time = trig_time + MEASURE_DURATION_MS / 1000.0
                label_measure = f"計測 ({MEASURE_DURATION_MS}ms)" if i == 0 else None
                ax.axvspan(trig_time, measure_end_time, color="yellow", alpha=0.2, label=label_measure)
                
                # ===== Peak Analysis logic =====
                # 500ms window for analysis
                analyze_window = 0.5 
                window_df = df[(df["time_rel"] >= trig_time) & (df["time_rel"] <= trig_time + analyze_window)]
                
                peak_val = 0.0
                peak_time_rel = 0.0
                
                if not window_df.empty:
                    # check peak in trigger channel (or all DOWN channels?)
                    # Usually we care about the channel that triggered, or the max of DOWN_CH
                    # Let's check max of DOWN_CH_IDX
                    vals = []
                    for d_idx in DOWN_CH_IDX:
                        c_name = f"ch{d_idx+1}"
                        if c_name in window_df.columns:
                            # 期間内の最大値を見つける
                            mx = window_df[c_name].max()
                            # その時刻
                            idxmax = window_df[c_name].idxmax()
                            t_mx = window_df.loc[idxmax, "time_rel"]
                            vals.append((mx, t_mx))
                    
                    # 最も高い値を持つチャンネルを採用
                    if vals:
                        vals.sort(key=lambda x: x[0], reverse=True)
                        peak_val, peak_abs_time = vals[0]
                        peak_time_rel = (peak_abs_time - trig_time) * 1000 # ms
                
                print(f"  [{i+1}] Time: {trig_time:.3f}s (ch: {trig_ch}) | Peak: {peak_val:.2f} (at {peak_time_rel:.1f}ms)")

        # 判定ライン (Strong/Weak) の再構築と表示
        # Stateの表示は削除し、こちらを表示
        decision_vals = np.zeros(len(df))
        
        # タイムスタンプからインデックスへのマッピング用
        times = df["time_rel"].values
        
        for i, (trig_time, trig_ch) in enumerate(trigger_events):
             # ピーク再計算 (判定用)
             analyze_window = 0.5
             window_df = df[(df["time_rel"] >= trig_time) & (df["time_rel"] <= trig_time + analyze_window)]
             
             peak_val = 0.0
             if not window_df.empty:
                 vals = []
                 for d_idx in DOWN_CH_IDX:
                     c_name = f"ch{d_idx+1}"
                     if c_name in window_df.columns:
                         vals.append(window_df[c_name].max())
                 if vals:
                     peak_val = max(vals)
             
             # 判定 (Thresholdは 0.3 ウェイトで計算済み)
             judge_val = -150 # Weak
             if peak_val > strong_threshold:
                 judge_val = -250 # Strong
             
             # インデックス範囲を取得 (numpy searchsortedが高速)
             start_idx = np.searchsorted(times, trig_time)
             end_idx = np.searchsorted(times, trig_time + 0.5) # 0.5秒間表示
             
             # 配列を埋める
             if start_idx < len(decision_vals):
                end_idx = min(end_idx, len(decision_vals))
                decision_vals[start_idx:end_idx] = judge_val
        
        # 判定ラインのプロット
        # 判定ラインのプロット
        ax.plot(df.loc[plot_mask, "time_rel"], decision_vals[plot_mask], label="判定 (50=弱 / 100=強)", color="magenta", linewidth=2.0)

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
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc="upper left")

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
