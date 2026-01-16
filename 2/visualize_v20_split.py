import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np

# 日本語フォント設定 (WindowsならMS Gothicが標準的)
plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["font.size"] = 14  # 文字サイズ調整


def select_file(root):
    # 初期ディレクトリを gitPython/9/CSV にしておくと便利
    initial_dir = r"G:\マイドライブ\GooglePython\csv"
    if not os.path.exists(initial_dir):
        initial_dir = os.getcwd()

    file_path = filedialog.askopenfilename(parent=root, initialdir=initial_dir, title="CSVファイルを選択してください", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
    return file_path


def select_channels(root):
    root.deiconify()  # ウィンドウを表示
    root.title("データ選択")

    # ウィンドウを画面中央に配置
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    w, h = 300, 500
    x = (sw - w) // 2
    y = (sh - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")

    # メインフレーム
    main_frame = tk.Frame(root)
    main_frame.pack(padx=20, pady=20, fill="both", expand=True)

    # === EMG Section ===
    tk.Label(main_frame, text="【EMG Channels】", font=("MS Gothic", 12, "bold")).pack(anchor="w", pady=(0, 5))

    emg_vars = []
    # チェックボックス作成 (ch1-ch8)
    for i in range(8):
        v = tk.BooleanVar(value=False)
        emg_vars.append(v)
        c = tk.Checkbutton(main_frame, text=f"Channel {i+1}", variable=v, font=("MS Gothic", 12))
        c.pack(anchor="w", padx=10, pady=1)

    # === Vibration Section ===
    tk.Label(main_frame, text="【Vibration Sensors】", font=("MS Gothic", 12, "bold")).pack(anchor="w", pady=(15, 5))

    vib_vars = []
    vib_names = ["vib1_z", "vib2_z"]
    for name in vib_names:
        v = tk.BooleanVar(value=False)
        vib_vars.append(v)
        c = tk.Checkbutton(main_frame, text=name, variable=v, font=("MS Gothic", 12))
        c.pack(anchor="w", padx=10, pady=1)

    # === Options ===
    tk.Label(main_frame, text="【Options】", font=("MS Gothic", 12, "bold")).pack(anchor="w", pady=(15, 5))

    measure_var = tk.BooleanVar(value=True)  # デフォルトON
    c_measure = tk.Checkbutton(main_frame, text="Manual Measurement Mode", variable=measure_var, font=("MS Gothic", 12))
    c_measure.pack(anchor="w", padx=10, pady=1)

    selected_emg = []
    selected_vib = []
    opts = {"manual_measure": True}

    def on_ok():
        for i, v in enumerate(emg_vars):
            if v.get():
                selected_emg.append(i)

        for i, v in enumerate(vib_vars):
            if v.get():
                selected_vib.append(vib_names[i])

        opts["manual_measure"] = measure_var.get()

        root.quit()  # mainloopを抜ける

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)

    btn = tk.Button(btn_frame, text="OK", command=on_ok, width=10, font=("MS Gothic", 12))
    btn.pack()

    root.mainloop()

    # 何も選択されなかった場合は全部表示 (デフォルト動作)
    if not selected_emg and not selected_vib:
        return list(range(8)), vib_names, opts["manual_measure"]

    return selected_emg, selected_vib, opts["manual_measure"]


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

    selected_ch_indices, selected_vib_cols, enable_manual_measure = select_channels(root)
    print(f"Selected Channels: {[i+1 for i in selected_ch_indices]}")
    print(f"Selected Vibrations: {selected_vib_cols}")
    print(f"Manual Measurement: {'ON' if enable_manual_measure else 'OFF'}")

    try:
        root.destroy()  # GUI終了
    except Exception:
        pass

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

        # V19のパラメータ (Z-score版)
        DOWN_CH_IDX = [5, 6]  # ch6, ch7
        STRONG_RATIO = 5.0
        # CALIB_DURATION = 5.0
        K_SIGMA = 5.7  # Trigger Threshold (Sigma)
        MEASURE_DURATION_MS = 140  # Measurement window for Strong/Weak check
        COOLDOWN_MS = 500  # Cooldown period

        # Vibration Sensor Constants (KXR94-2050)
        VIB_V_REF = 5.0  # Arduino VCC
        VIB_SENSITIVITY = 0.660  # V/g
        VIB_ZERO_G_ADC = 470  # Zero-g offset (approx 1.65V, adjusted)

        calib_rest_end = 5.0
        calib_weak_start = 8.0
        calib_weak_end = 11.0
        calib_strong_start = 14.0
        calib_strong_end = 17.0
        base = 18

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
                    if cal_stds[i] < 1e-6:
                        cal_stds[i] = 1e-6
        else:
            print("Warning: No Resting Data found.")

        # Trigger Thresholds (Base)
        trigger_thresholds = cal_means + cal_stds * K_SIGMA
        print(f"Trigger Thresholds (Mean + {K_SIGMA}sigma): {trigger_thresholds}")

        # 2. Weak Calibration (8-11s)
        weak_df = df[(df["time_rel"] >= calib_weak_start) & (df["time_rel"] <= calib_weak_end)]
        weak_max = 0.0
        if not weak_df.empty:
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
            vals = []
            for idx in DOWN_CH_IDX:
                vals.append(cal_means[idx] + cal_stds[idx] * STRONG_RATIO)
            if vals:
                strong_threshold = np.mean(vals)
            print("Warning: Using default Strong Threshold (no strong calib data).")

        print(f"Strong Threshold: {strong_threshold:.2f}")

        # グラフ作成 (Change 1: Split to 2 subplots)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        plt.subplots_adjust(hspace=0.1)  # 上下の隙間を詰める

        judgment_artists = []  # Toggle 用のリスト

        # === 上段: EMGデータ (ch1-ch8) ===
        emg_palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#8c564b", "#9467bd", "#e377c2", "#7f7f7f"]

        for i in selected_ch_indices:
            col = f"ch{i+1}"
            if col in df.columns:
                color_idx = i % len(emg_palette)
                c = emg_palette[color_idx]

                lines = ax1.plot(df["time_rel"], df[col], label=f"筋電位 {col}", alpha=1.0, linewidth=1.5, color=c)
                line_color = lines[0].get_color()

                # Trigger Threshold
                t_thr = trigger_thresholds[i]
                if t_thr > 0:
                    # Restore label
                    thresh_line = ax1.axhline(y=t_thr, color=line_color, linestyle="--", linewidth=1.0, alpha=0.7, label=f"{col} 閾値 ({t_thr:.1f})")
                    judgment_artists.append(thresh_line)

        # Strong Threshold (共通)
        is_down_ch_selected = any(idx in DOWN_CH_IDX for idx in selected_ch_indices)
        if is_down_ch_selected:
            thresh_line = ax1.axhline(y=strong_threshold, color="black", linestyle="--", linewidth=1.5)  # Remove label
            judgment_artists.append(thresh_line)

            # s_anno_x = calib_strong_end + 3.0 (Unused)

            thresh_line = ax1.axhline(y=strong_threshold, color="black", linestyle="--", linewidth=1.5, label=f"強閾値 ({strong_threshold:.1f})")
            judgment_artists.append(thresh_line)

        # === 下段: 振動データ (vib1_z, vib2_z) ===
        vib_color_map = {"vib1_z": "#1f77b4", "vib2_z": "#ff7f0e"}
        vib_label_map = {"vib1_z": "振動 腕", "vib2_z": "振動 義手"}
        for col in selected_vib_cols:
            if col in df.columns:
                # Convert to voltage then to g
                # g = (ADC - Zero_G) * (V_Ref / 1023) / Sensitivity
                v_data = (df[col] - VIB_ZERO_G_ADC) * (VIB_V_REF / 1023.0) / VIB_SENSITIVITY

                c = vib_color_map.get(col, "black")
                display_name = vib_label_map.get(col, col)
                ax2.plot(df["time_rel"], v_data, label=display_name, linestyle="-", linewidth=1.5, alpha=0.8, color=c)

        # ===== Latency Analysis & Judgment =====

        # Calibration Windows Visualization (ax1 only or both? Let's put on ax1)
        # Calibration Windows Visualization (ax1 only or both? Let's put on ax1)
        l1 = ax1.axvspan(calib_weak_start, calib_weak_end, color="blue", alpha=0.1)
        l2 = ax1.axvspan(calib_strong_start, calib_strong_end, color="red", alpha=0.1)
        judgment_artists.extend([l1, l2])

        analysis_df = df[df["time_rel"] > calib_strong_end].copy()
        trigger_events = []
        next_available_time = 0.0

        if not analysis_df.empty and not rest_df.empty:
            for idx in range(len(analysis_df)):
                row_idx = analysis_df.index[idx]
                t = analysis_df["time_rel"].loc[row_idx]

                if t < next_available_time:
                    continue

                for ch_idx in DOWN_CH_IDX:
                    col = f"ch{ch_idx+1}"
                    if col in df.columns:
                        val = df[col].loc[row_idx]
                        mean_val = cal_means[ch_idx]
                        std_val = cal_stds[ch_idx]
                        if (val - mean_val) / std_val > K_SIGMA:
                            trigger_events.append((t, col))
                            next_available_time = t + (MEASURE_DURATION_MS + COOLDOWN_MS) / 1000.0
                            break

        if trigger_events:
            print(f"Detected {len(trigger_events)} triggers.")
            for i, (trig_time, trig_ch) in enumerate(trigger_events):
                label_name = "トリガー" if i == 0 else None

                # Trigger Line (Vertical) - Draw on both axes
                l1 = ax1.axvline(x=trig_time, color="orange", linestyle="-.", linewidth=1.5, label=label_name)
                l2 = ax2.axvline(x=trig_time, color="orange", linestyle="-.", linewidth=1.5)
                judgment_artists.extend([l1, l2])

                # Annotation for Trigger

                # Measurement Duration (Shaded Area)
                measure_end_time = trig_time + MEASURE_DURATION_MS / 1000.0
                label_measure = f"計測 ({MEASURE_DURATION_MS}ms)" if i == 0 else None
                s1 = ax1.axvspan(trig_time, measure_end_time, color="yellow", alpha=0.2, label=label_measure)
                s2 = ax2.axvspan(trig_time, measure_end_time, color="yellow", alpha=0.2)
                judgment_artists.extend([s1, s2])

                # Peak logic... (just for print)
                analyze_window = 0.5
                window_df = df[(df["time_rel"] >= trig_time) & (df["time_rel"] <= trig_time + analyze_window)]
                peak_val = 0.0
                peak_time_rel = 0.0
                if not window_df.empty:
                    vals = []
                    for d_idx in DOWN_CH_IDX:
                        c_name = f"ch{d_idx+1}"
                        if c_name in window_df.columns:
                            mx = window_df[c_name].max()
                            idxmax = window_df[c_name].idxmax()
                            t_mx = window_df.loc[idxmax, "time_rel"]
                            vals.append((mx, t_mx))
                    if vals:
                        vals.sort(key=lambda x: x[0], reverse=True)
                        peak_val, peak_abs_time = vals[0]
                        peak_time_rel = (peak_abs_time - trig_time) * 1000
                print(f"  [{i+1}] Time: {trig_time:.3f}s (ch: {trig_ch}) | Peak: {peak_val:.2f} (at {peak_time_rel:.1f}ms)")

        # 判定ライン (Strong/Weak)
        decision_vals = np.zeros(len(df))
        times = df["time_rel"].values
        for i, (trig_time, trig_ch) in enumerate(trigger_events):
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

            judge_val = -50
            if peak_val > strong_threshold:
                judge_val = -80

            start_idx = np.searchsorted(times, trig_time)
            end_idx = np.searchsorted(times, trig_time + 0.5)
            if start_idx < len(decision_vals):
                end_idx = min(end_idx, len(decision_vals))
                decision_vals[start_idx:end_idx] = judge_val

        # Plot decision on ax1 (EMG plot)
        # (l_dec,) = ax1.plot(df["time_rel"], decision_vals, label="判定 (50=弱 / 100=強)", color="magenta", linewidth=2.0)
        # judgment_artists.append(l_dec)

        # Titles and Labels
        # ax1.set_title("センサーデータ可視化 (上: 筋電 / 下: 振動)", fontsize=16)
        ax1.set_ylabel("筋電位", fontsize=18)
        ax2.set_ylabel("振動(g)", fontsize=18)

        ax2.set_xlabel("時間 (秒)", fontsize=18)

        ax1.tick_params(labelsize=12)
        ax2.tick_params(labelsize=12)
        ax1.grid(True)
        ax2.grid(True)

        # Legends
        ax1.legend(fontsize=22, loc="upper left")
        ax2.legend(fontsize=22, loc="upper left")

        # ===== Key Press Event for Toggle =====
        def on_key(event):
            if event.key == "d" or event.key == "D":
                # Toggle visibility
                new_visible = not judgment_artists[0].get_visible() if judgment_artists else True
                for art in judgment_artists:
                    art.set_visible(new_visible)
                vis_str = "ON" if new_visible else "OFF"
                print(f"[Toggle] Judgment Lines: {vis_str}")
                fig.canvas.draw()

        fig.canvas.mpl_connect("key_press_event", on_key)

        # ===== Interactive Manual Measurement =====
        click_state = {"count": 0, "start_time": None, "start_val": None, "markers": []}

        def onclick(event):
            if event.inaxes not in [ax1, ax2]:
                return
            if event.button != 1:
                return

            t = event.xdata
            y = event.ydata
            current_ax = event.inaxes

            click_state["count"] += 1
            count = click_state["count"]

            if count == 1:
                click_state["start_time"] = t
                click_state["start_val"] = y
                for m in click_state["markers"]:
                    try:
                        m.remove()
                    except:
                        pass
                click_state["markers"] = []

                (marker,) = current_ax.plot(t, y, "go", markersize=8, zorder=10)
                click_state["markers"].append(marker)
                print(f"[Manual] Start Point: {t:.3f} s")
                plt.draw()

            elif count == 2:
                start_t = click_state["start_time"]
                diff_ms = abs(t - start_t) * 1000
                print(f"[Manual] End Point:   {t:.3f} s")
                print(f"[Manual] Diff:        {diff_ms:.1f} ms")

                ann = current_ax.annotate(
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
                click_state["count"] = 1
                for m in click_state["markers"]:
                    try:
                        m.remove()
                    except:
                        pass
                click_state["markers"] = []
                click_state["start_time"] = t
                click_state["start_val"] = y
                (marker,) = current_ax.plot(t, y, "go", markersize=8, zorder=10)
                click_state["markers"].append(marker)
                print(f"[Manual] Start Point (Reset): {t:.3f} s")
                plt.draw()

        if enable_manual_measure:
            fig.canvas.mpl_connect("button_press_event", onclick)

        plt.tight_layout()
        ax1.set_xlim(left=base, right=df["time_rel"].max())
        plt.show()

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
