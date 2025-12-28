import time
import numpy as np
from collections import deque
from pyomyo import Myo, emg_mode
import serial
import threading

# ===== Arduino設定 =====
SERIAL_MOTOR = "COM6"  # ★環境に合わせて変更
BAUDRATE = 115200

try:
    ser_motor = serial.Serial(SERIAL_MOTOR, BAUDRATE, timeout=0.01)
    print(f"Arduino connected: {SERIAL_MOTOR}")
except:
    ser_motor = None

# ===== 制御パラメータ =====
UP_CH = [1, 2]  # 振り上げ (配列index 0-7)
DOWN_CH = [5, 6]  # 攻撃

MEASURE_DURATION_MS = 50
COOLDOWN_MS = 150

# 感度設定 (変化量に対する倍率)
# ノイズの変化量の何倍の急上昇で反応するか
RISE_SENSITIVITY = 4

# UP動作は「維持」が必要なので、絶対値判定用の倍率
UP_HOLD_SENSITIVITY = 5.0

STRONG_RATIO = 1.5
EMA_ALPHA = 0.3

# ===== グローバル変数 =====
rms_buf = [deque(maxlen=30) for _ in range(8)]
ema_val = np.zeros(8)
prev_ema_val = np.zeros(8)  # 前回の値を保存（微分用）

# 閾値格納用
rise_thresholds = np.zeros(8) + 999.0  # 変化量の閾値 (DOWN用)
level_thresholds = np.zeros(8) + 999.0  # 絶対値の閾値 (UPホールド用)
strong_threshold = 999.0

calibration_done = False

# 状態管理
STATE_IDLE = 0
STATE_ATTACKING = 1
STATE_COOLDOWN = 2

current_state = STATE_IDLE
measure_start_time = 0
cooldown_start_time = 0
measure_buffer = []
is_holding_up = False


# ===== 関数 =====
def compute_rms(buf):
    if len(buf) == 0:
        return 0
    return np.sqrt(np.mean(np.array(buf) ** 2))


def send_cmd(cmd):
    if ser_motor and ser_motor.is_open:
        try:
            ser_motor.write(cmd.encode("utf-8"))
        except:
            pass


def calibrate(duration=3.0):
    """
    静止時の「値」と「変化量」の両方を測定する
    """
    global rise_thresholds, level_thresholds, strong_threshold, calibration_done
    print("\n=== キャリブレーション: 3秒間 脱力してください ===")
    time.sleep(duration)

    val_samples = []  # 値そのもの
    diff_samples = []  # 変化量 (今回 - 前回)

    print("... ノイズ学習中 ...")

    # ダミーデータを数回回して prev_ema_val を安定させる
    for _ in range(10):
        time.sleep(0.01)

    for _ in range(50):
        current_val = np.copy(ema_val)
        current_diff = np.abs(ema_val - prev_ema_val)  # 変化の絶対値

        val_samples.append(current_val)
        diff_samples.append(current_diff)
        time.sleep(0.01)

    # 1. UP用: 絶対値のノイズレベル (Mean + Std * K)
    val_mean = np.mean(val_samples, axis=0)
    val_std = np.std(val_samples, axis=0)
    level_thresholds = val_mean + (val_std * UP_HOLD_SENSITIVITY)
    level_thresholds = np.maximum(level_thresholds, 15.0)

    # 2. DOWN用: 変化量のノイズレベル (Diff_Mean + Diff_Std * K)
    # これが「急激な上昇」の基準になります
    diff_mean = np.mean(diff_samples, axis=0)
    diff_std = np.std(diff_samples, axis=0)
    rise_thresholds = diff_mean + (diff_std * RISE_SENSITIVITY)

    # 安全マージン（変化量が小さすぎると少しのノイズで反応するため）
    rise_thresholds = np.maximum(rise_thresholds, 2.0)

    # 3. 強打判定用 (値の大きさで見る)
    # 攻撃開始は「変化量」で見ますが、強さは「到達した値」で見ます
    strong_threshold = np.mean(level_thresholds[DOWN_CH]) * STRONG_RATIO

    calibration_done = True
    print("=== 完了 ===")
    print(f"UP Hold Thr (Level): {np.round(level_thresholds[UP_CH], 1)}")
    print(f"DOWN Attack Thr (Slope): {np.round(rise_thresholds[DOWN_CH], 1)}")


def on_emg(emg, movement):
    global current_state, measure_start_time, cooldown_start_time, measure_buffer
    global is_holding_up, ema_val, prev_ema_val

    if emg is None:
        return

    # 値の更新
    prev_ema_val = np.copy(ema_val)  # 前回の値を退避

    for ch in range(8):
        rms_buf[ch].append(emg[ch])
        rms = compute_rms(rms_buf[ch])
        ema_val[ch] = EMA_ALPHA * rms + (1 - EMA_ALPHA) * ema_val[ch]

    if not calibration_done:
        return

    # --- 特徴量の計算 ---

    # UP: 絶対値を見る (ホールドするため)
    up_level = np.max([ema_val[ch] for ch in UP_CH])

    # DOWN: 変化量(傾き)を見る (瞬発力検出)
    # (現在値 - 前回値) が正の方向に急上昇したか？
    down_slope_list = [(ema_val[ch] - prev_ema_val[ch]) for ch in DOWN_CH]
    max_down_slope = np.max(down_slope_list)

    # 強弱判定用に絶対値も取っておく
    down_level = np.max([ema_val[ch] for ch in DOWN_CH])

    now = time.time()

    # --- ステートマシン ---

    if current_state == STATE_IDLE:

        # [判定 A] 攻撃トリガー (急激な上昇検出)
        # 傾き(slope) が 閾値(rise_thresholds) を超えたら発動
        is_attack_triggered = False
        for i, ch in enumerate(DOWN_CH):
            if down_slope_list[i] > rise_thresholds[ch]:
                is_attack_triggered = True
                break

        if is_attack_triggered:
            # ★見切り発車 ('I')
            send_cmd("I")

            current_state = STATE_ATTACKING
            measure_start_time = now
            measure_buffer = []
            is_holding_up = False
            print(f">> ATTACK! (Slope: {max_down_slope:.2f})")
            return

        # [判定 B] 振り上げ (絶対値判定)
        # 上げ動作は「維持」が必要なので、従来どおりレベル(絶対値)で見る
        is_up_active = up_level > np.mean(level_thresholds[UP_CH])

        if is_up_active and not is_holding_up:
            send_cmd("L")
            is_holding_up = True

        elif not is_up_active and is_holding_up:
            send_cmd("R")
            is_holding_up = False

    elif current_state == STATE_ATTACKING:
        measure_buffer.append(down_level)  # 強弱判定は「値の大きさ」で行う

        if (now - measure_start_time) * 1000 >= MEASURE_DURATION_MS:
            if len(measure_buffer) > 0:
                # ピーク値を判定に使う（平均よりピークの方が打撃感に近い）
                peak_val = np.max(measure_buffer)

                if peak_val > strong_threshold:
                    send_cmd("S")
                    print(f"[HIT] STRONG (Level:{peak_val:.1f})")
                else:
                    send_cmd("W")
                    print(f"[HIT] weak   (Level:{peak_val:.1f})")

            current_state = STATE_COOLDOWN
            cooldown_start_time = now

    elif current_state == STATE_COOLDOWN:
        if (now - cooldown_start_time) * 1000 >= COOLDOWN_MS:
            send_cmd("R")
            current_state = STATE_IDLE
            is_holding_up = False


# ===== メイン処理 =====
def main():
    print("Myo接続中...")
    m = Myo(mode=emg_mode.RAW)
    m.connect()
    m.add_emg_handler(on_emg)

    # スレッド開始
    t = threading.Thread(target=lambda: m.run(), daemon=True)
    t.start()

    m.set_leds([0, 255, 255], [0, 255, 255])
    m.vibrate(1)
    send_cmd("R")

    calibrate()

    print("\n==== 立ち上がり検出モード ====")
    print(" 攻撃(5,6ch): 急激に力を入れると反応します")
    print(" 振り上げ(1,2ch): 力を入れている間上がります")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        m.disconnect()
        if ser_motor:
            ser_motor.close()


if __name__ == "__main__":
    main()