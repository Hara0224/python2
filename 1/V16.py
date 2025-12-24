import time
import numpy as np
from collections import deque
from pyomyo import Myo, emg_mode
import serial
import threading

# ===== Arduino設定 =====
SERIAL_PORT = "COM5"
BAUDRATE = 115200

# ===== 制御パラメータ =====
MEASURE_DURATION_MS = 50
COOLDOWN_MS = 150
TRIGGER_MULTIPLIER = 5.0
STRONG_RATIO = 1.5

# チャンネル設定
UP_CH = [1, 2]  # 振り上げ用 (伸筋)
DOWN_CH = [5, 6]  # 振り下ろし/攻撃用 (屈筋)

# ===== 変数初期化 =====
rms_buf = [deque(maxlen=30) for _ in range(8)]
ema_val = np.zeros(8)
trigger_thresholds = np.zeros(8) + 999.0
strong_threshold = 999.0

# 状態定数
STATE_IDLE = 0  # 待機 (下がっている、または上げている途中)
STATE_ATTACKING = 1  # 攻撃判定中 (見切り発車中)
STATE_COOLDOWN = 2  # 攻撃完了後

current_state = STATE_IDLE
measure_start_time = 0
cooldown_start_time = 0
measure_buffer = []
is_holding_up = False  # 状態フラグ: 上げているかどうか

calibration_done = False
EMA_ALPHA = 0.3

# ===== シリアル通信 =====
try:
    ser_motor = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.01)
except:
    ser_motor = None


def send_command(cmd):
    if ser_motor:
        try:
            ser_motor.write(cmd.encode("utf-8"))
        except:
            pass


# ===== 計算・キャリブレーション =====
def compute_rms(buf):
    if len(buf) == 0:
        return 0
    return np.sqrt(np.mean(np.array(buf) ** 2))


def calibrate(duration=3.0):
    global trigger_thresholds, strong_threshold, calibration_done
    print(f"=== キャリブレーション: {duration}秒間 脱力してください ===")
    time.sleep(duration)

    samples = []
    for _ in range(50):
        samples.append(np.copy(ema_val))
        time.sleep(0.01)

    noise_mean = np.mean(samples, axis=0)
    noise_std = np.std(samples, axis=0)
    trigger_thresholds = noise_mean + (noise_std * TRIGGER_MULTIPLIER)
    trigger_thresholds = np.maximum(trigger_thresholds, 15.0)

    # 強打判定値
    base_trigger = np.mean(trigger_thresholds[DOWN_CH])
    strong_threshold = base_trigger * STRONG_RATIO

    calibration_done = True
    print(f"Trigger Thr (UP): {np.round(trigger_thresholds[UP_CH], 1)}")
    print(f"Trigger Thr (DOWN): {np.round(trigger_thresholds[DOWN_CH], 1)}")


# ===== メインロジック =====
def on_emg(emg, movement):
    global current_state, measure_start_time, cooldown_start_time
    global measure_buffer, is_holding_up, ema_val

    if emg is None:
        return

    for ch in range(8):
        rms_buf[ch].append(emg[ch])
        rms = compute_rms(rms_buf[ch])
        ema_val[ch] = EMA_ALPHA * rms + (1 - EMA_ALPHA) * ema_val[ch]

    if not calibration_done:
        return

    # 強度取得
    up_val = np.max([ema_val[ch] for ch in UP_CH])
    down_val = np.max([ema_val[ch] for ch in DOWN_CH])
    now = time.time()

    # --- ステートマシン ---

    # [状態: 待機または振り上げ中]
    if current_state == STATE_IDLE:

        # 優先順位1: 攻撃 (DOWN)
        # 振り上げ中でも、DOWNに力が入ったら即攻撃に移行する
        if down_val > trigger_thresholds[DOWN_CH].mean():
            # ★攻撃開始 (Initial Dash -> Down)
            send_command("I")
            current_state = STATE_ATTACKING
            measure_start_time = now
            measure_buffer = []
            is_holding_up = False  # 攻撃したらHold解除
            # print(">> ATTACK START")
            return

        # 優先順位2: 振り上げ (UP)
        # 攻撃していない時だけ有効
        up_triggered = up_val > trigger_thresholds[UP_CH].mean()

        if up_triggered and not is_holding_up:
            # 上げる命令 ('L')
            send_command("L")
            is_holding_up = True
            # print("^^ LIFT")

        elif not up_triggered and is_holding_up:
            # 力を抜いたら下げる ('R')
            # これを入れないと上がりっぱなしになります
            send_command("R")
            is_holding_up = False
            # print("__ DROP")

    # [状態: 攻撃判定中 (見切り発車中)]
    elif current_state == STATE_ATTACKING:
        measure_buffer.append(down_val)

        if (now - measure_start_time) * 1000 >= MEASURE_DURATION_MS:
            if len(measure_buffer) > 0:
                avg = np.mean(measure_buffer)

                if avg > strong_threshold:
                    send_command("S")  # 強打 (Down Max Speed)
                    print(f"[HIT] STRONG ({avg:.1f})")
                else:
                    send_command("W")  # 弱打 (Brake)
                    print(f"[HIT] weak   ({avg:.1f})")

            current_state = STATE_COOLDOWN
            cooldown_start_time = now

    # [状態: クールダウン]
    elif current_state == STATE_COOLDOWN:
        if (now - cooldown_start_time) * 1000 >= COOLDOWN_MS:
            # 戻り処理 ('R')
            # 攻撃後は必ず一度下に戻る
            send_command("R")
            current_state = STATE_IDLE
            is_holding_up = False


# ===== メイン処理 =====
def main():
    m = Myo(mode=emg_mode.RAW)
    m.connect()
    m.add_emg_handler(on_emg)

    def worker():
        while True:
            m.run()

    threading.Thread(target=worker, daemon=True).start()

    m.set_leds([0, 255, 0], [0, 255, 0])
    m.vibrate(1)

    # 起動時リセット
    send_command("R")

    try:
        calibrate()
        print("\n==== DRUM CONTROL READY ====")
        print(" UP(2,3ch)   -> Lift")
        print(" DOWN(5,6ch) -> Attack (Variable Speed)")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        m.disconnect()
        if ser_motor:
            ser_motor.close()


if __name__ == "__main__":
    main()
