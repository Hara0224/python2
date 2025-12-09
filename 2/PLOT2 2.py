import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import os

# =========================================================
# ===== 判定ロジックの定数 (MyoControllerクラスから抽出) =====
# =========================================================

CSV_PATH = r"C:\Users\hrsyn\Desktop\Python\emg_data3\emg_rms1.csv"  # RMSデータファイル名
CALIB_DURATION = 3.0  # キャリブレーション秒数
FS = 200.0  # サンプリング周波数 (Hz)

# 判定に必要な主要パラメータ
K_SIGMA = 2.7  # 閾値計算の標準偏差倍率 (mean + K_SIGMA * std = 100% Threshold)
PEAK_DELTA = 6.0  # RMS変化量のトリガ閾値 (Delta >= PEAK_DELTA)
START_MARGIN = 0.5  # 判定開始点 (0.5閾値)
END_MARGIN = 0.8  # 確実性トリガの閾値 (1.0閾値)
PEAK_CEILING_MARGIN = 1.2  # 急峻トリガの絶対上限点 (1.2閾値)
TRANSITION_LOOKBACK = 5  # 確実性トリガで低レベル移行をチェックするサンプル数
START_MARGIN_TOLERANCE = 0.1  # 低レベル移行チェックの許容誤差

# トリガ対象チャンネル
up_ch_idx = [2 - 1, 3 - 1]  # CH2 (インデックス1), CH3 (インデックス2)
down_ch_idx = [7 - 1]  # CH7 (インデックス6)
target_channels = sorted(list(set(up_ch_idx + down_ch_idx)))

# =========================================================
# ===== データ読み込みとキャリブレーションのシミュレーション =====
# =========================================================

# --- 修正箇所: dfの定義をtry-exceptブロックで確実にする ---
try:
    # 実際のデータ読み込み
    df = pd.read_csv(CSV_PATH)
    print(f"データ: {CSV_PATH} を正常に読み込みました。")

except FileNotFoundError:
    # ファイルが見つからない場合、デバッグ用のダミーデータを生成
    print(f"エラー: {CSV_PATH} が見つかりません。デバッグモード: 仮のダミーデータを生成して続行します。")
    time_s = np.linspace(0, 10, int(10 * FS), endpoint=False)
    num_samples = len(time_s)
    n_ch = 8
    emg_rms = np.zeros((num_samples, n_ch))

    # チャンネル2と7に活動をシミュレーション
    emg_rms[:, 1] = 5 + 1 * np.random.randn(num_samples)  # CH2 (Index 1)
    emg_rms[:, 6] = 5 + 1 * np.random.randn(num_samples)  # CH7 (Index 6)

    # 活動を急激に増加させる (約3秒、7秒時点)
    start_idx_1 = int(3 * FS)
    end_idx_1 = int(3.5 * FS)
    emg_rms[start_idx_1:end_idx_1, 1] += np.linspace(0, 50, end_idx_1 - start_idx_1)
    emg_rms[start_idx_1:end_idx_1, 1] += 50

    start_idx_2 = int(7 * FS)
    end_idx_2 = int(7.5 * FS)
    emg_rms[start_idx_2:end_idx_2, 6] += np.linspace(0, 60, end_idx_2 - start_idx_2)
    emg_rms[start_idx_2:end_idx_2, 6] += 40

    # ここでdfを定義
    column_names = ["Time"] + [f"CH{i+1}" for i in range(n_ch)]
    df = pd.DataFrame(np.hstack([time_s.reshape(-1, 1), emg_rms]), columns=column_names)

# --- dfが定義されていることを保証した後、値を抽出 ---
time_s = df.iloc[:, 0].values
emg_rms = df.iloc[:, 1:].values  # RMSデータ
n_ch = emg_rms.shape[1]

# 1. キャリブレーションのシミュレーション
calib_samples = int(CALIB_DURATION * FS)
if calib_samples >= len(emg_rms):
    calib_samples = len(emg_rms) // 2
    print(f"[WARN] データ長が短いため、最初の {calib_samples} サンプルでキャリブレーションを代替しました。")

calib_data = emg_rms[:calib_samples, :]
mean = calib_data.mean(axis=0)
std = calib_data.std(axis=0, ddof=1)
std[std < 1e-6] = 1e-6

# =========================================================
# ===== 判定ロジックの適用とトリガ領域の特定 (MyoController準拠) =====
# =========================================================

# 2. 基本閾値 (Threshold: 100%ライン) の計算
thresholds = mean + K_SIGMA * std

# 3. RMS変化量 (Delta) の計算
delta_emg = emg_rms - np.roll(emg_rms, 1, axis=0)
delta_emg[0, :] = 0

# 4. 判定レベルの計算
start_point = thresholds * START_MARGIN  # 0.5T (下限)
end_point = thresholds * END_MARGIN  # 1.0T (確実性トリガの閾値)
ceiling_point = thresholds * PEAK_CEILING_MARGIN  # 1.2T (絶対上限)
low_activity_check_point = thresholds * (START_MARGIN + START_MARGIN_TOLERANCE)  # 0.6T (移行チェック用)

# 5. 各サンプルのトリガ条件判定
num_samples = len(emg_rms)
steep_trigger_area = np.zeros_like(emg_rms, dtype=bool)
certainty_trigger_area = np.zeros_like(emg_rms, dtype=bool)

# 各チャンネルと各サンプルについてループ
for ch_idx in target_channels:

    # 確実性トリガ判定用の履歴リスト（RMS履歴をシミュレート）
    rms_history_ch = emg_rms[:, ch_idx]

    for t in range(num_samples):
        rms_now = rms_history_ch[t]
        delta = delta_emg[t, ch_idx]

        # --- (A) Steep Trigger (急峻トリガ) ---
        is_steep_trigger = (rms_now >= start_point[ch_idx]) and (rms_now < ceiling_point[ch_idx]) and (delta >= PEAK_DELTA)  # 1. START_MARGIN以上  # 2. CEILING_MARGIN未満  # 3. PEAK_DELTA以上の傾き

        # --- (B) Certainty Trigger (確実性トリガ) ---
        is_certainty_trigger = False
        is_over_end_point = rms_now >= end_point[ch_idx]  # 1. END_MARGIN以上

        if is_over_end_point:
            # 2. 低レベルからの移行チェック
            is_transition_from_low = False

            # 直近 TRANSITION_LOOKBACK 分の履歴をチェック (t=0, 1... はスキップ)
            start_lookback = max(0, t - TRANSITION_LOOKBACK)

            # 現在のサンプル(t)を含まない過去のウィンドウ
            lookback_window = rms_history_ch[start_lookback:t]

            # 履歴の中に、低い活動レベル（チェックポイント以下）のRMS値があれば True
            if any(rms_val <= low_activity_check_point[ch_idx] for rms_val in lookback_window):
                is_transition_from_low = True

            is_certainty_trigger = is_over_end_point and is_transition_from_low

        # --- 最終判定 ---
        if is_steep_trigger:
            steep_trigger_area[t, ch_idx] = True

        if is_certainty_trigger:
            certainty_trigger_area[t, ch_idx] = True


# 最終トリガ領域 (急峻 OR 確実性)
final_trigger_area = steep_trigger_area | certainty_trigger_area


# =========================================================
# ===== グラフ表示 (対象チャンネルのみ) =====
# =========================================================

# ターゲットチャンネル数に基づいてサブプロットを調整
fig, axes = plt.subplots(len(target_channels), 1, figsize=(12, 3.5 * len(target_channels)), sharex=True)
if len(target_channels) == 1:
    axes = [axes]

fig.suptitle("EMGトリガ判定ロジック可視化 (2種のトリガ条件)", fontsize=16)

for i, ch_idx in enumerate(target_channels):
    ax = axes[i]
    ch_num = ch_idx + 1
    direction = "UP" if ch_idx in up_ch_idx else "DOWN"

    # A. 通常のEMG信号を「線」ではなく「点（マーカー）」でプロット
    ax.plot(
        time_s,
        emg_rms[:, ch_idx],
        color="#2061A0",
        label="RMS Value (Sample Points)",
        linestyle="",  # 線を非表示
        marker=".",  # ドットのマーカーを使用
        markersize=4,  # マーカーのサイズを設定
    )

    # B. 閾値ライン
    ax.axhline(
        thresholds[ch_idx],
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"END Margin ({END_MARGIN*100:.0f}%)",
    )
    ax.axhline(
        start_point[ch_idx],
        color="orange",
        linestyle=":",
        linewidth=1,
        label=f"START Margin ({START_MARGIN*100:.0f}%)",
    )
    ax.axhline(
        ceiling_point[ch_idx],
        color="purple",
        linestyle="-.",
        linewidth=1,
        label=f"CEILING Margin ({PEAK_CEILING_MARGIN*100:.0f}%)",
    )
    ax.axhline(
        low_activity_check_point[ch_idx],
        color="gray",
        linestyle=":",
        linewidth=1,
        label=f"Low Activity Check ({START_MARGIN*100:.0f}% + {START_MARGIN_TOLERANCE*100:.0f}%)",
    )

    # C. 判定可能範囲の塗りつぶし (START〜CEILING)
    ax.axhspan(
        start_point[ch_idx],
        ceiling_point[ch_idx],
        color="lightgray",
        alpha=0.3,
        label=f"Steep Range ({START_MARGIN*100:.0f}%〜{PEAK_CEILING_MARGIN*100:.0f}%)",
    )

    # D. トリガ条件を満たした領域を色分けして強調

    # マゼンタ: 急峻トリガが発動した領域 (点として強調)
    steep_indices = np.where(steep_trigger_area[:, ch_idx])[0]
    if steep_indices.size > 0:
        split_idx = np.split(steep_indices, np.where(np.diff(steep_indices) != 1)[0] + 1)
        for s in split_idx:
            # 急峻トリガ領域の点を描画
            ax.plot(
                time_s[s],
                emg_rms[s, ch_idx],
                color="magenta",
                marker="o",
                markersize=5,
                linestyle="",  # 線は使わない
                label="Steep Trigger Area" if s is split_idx[0] else None,
            )

            # E. 最初の急峻トリガポイント (濃い紫の点で強調)
            ax.plot(
                time_s[s[0]],
                emg_rms[s[0], ch_idx],
                marker="o",
                markersize=8,
                color="darkviolet",
                linestyle="",
                zorder=5,
                label="Initial Steep Trigger" if s is split_idx[0] else None,
            )

    # シアン: 確実性トリガが発動した領域 (元々散布図なので変更なし)
    certainty_indices = np.where(certainty_trigger_area[:, ch_idx])[0]
    if certainty_indices.size > 0:
        split_idx = np.split(certainty_indices, np.where(np.diff(certainty_indices) != 1)[0] + 1)
        for s in split_idx:
            # Steep Triggerと重複しない純粋なCertainty Triggerの領域を抽出
            pure_certainty = s[~steep_trigger_area[s, ch_idx]]
            if pure_certainty.size > 0:
                ax.scatter(
                    time_s[pure_certainty],
                    emg_rms[pure_certainty, ch_idx],
                    color="cyan",
                    marker="^",
                    s=40,
                    zorder=4,
                    label="Certainty Trigger Point" if s is split_idx[0] else None,
                )

    ax.set_title(f"Channel {ch_num} ({direction} Trigger Channel) | Threshold={thresholds[ch_idx]:.2f}")
    ax.set_ylabel("RMS Value")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

axes[-1].set_xlabel("Time [s]")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# =========================================================
# ===== 補足：変化量 (Delta) の可視化 (オプション) =====
# =========================================================

# Deltaのプロットは引き続き線（Deltaの変化は線で見せる方が直感的）
fig_delta, axes_delta = plt.subplots(len(target_channels), 1, figsize=(12, 3.5 * len(target_channels)), sharex=True)
if len(target_channels) == 1:
    axes_delta = [axes_delta]

fig_delta.suptitle("RMS変化量 (Delta) の確認", fontsize=16)

for i, ch_idx in enumerate(target_channels):
    ax = axes_delta[i]
    ch_num = ch_idx + 1

    # Deltaのライン
    ax.plot(
        time_s,
        delta_emg[:, ch_idx],
        color="green",
        label="Delta (RMS_t - RMS_{t-1})",
        linewidth=1,
    )

    # PEAK_DELTAの閾値ライン
    ax.axhline(
        PEAK_DELTA,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"PEAK_DELTA ({PEAK_DELTA})",
    )
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)

    # 閾値を超えたDeltaを強調
    delta_over_thr = delta_emg[:, ch_idx].copy()
    delta_over_thr[delta_emg[:, ch_idx] < PEAK_DELTA] = np.nan
    ax.plot(time_s, delta_over_thr, color="red", linewidth=2, label="Delta >= PEAK_DELTA")

    ax.set_title(f"Channel {ch_num} Delta Value")
    ax.set_ylabel("Delta")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

axes_delta[-1].set_xlabel("Time [s]")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
