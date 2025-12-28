import time
import numpy as np
from collections import deque
from pyomyo import Myo, emg_mode
import serial
import threading
import sys

# ===== Arduino設定 =====
SERIAL_PORT = "COM5"  # ★環境に合わせて変更してください
BAUDRATE = 115200

# ===== 制御パラメータ =====
# 1145BWは高速なため、判定時間を短く(50ms)設定
MEASURE_DURATION_MS = 50
COOLDOWN_MS = 150  # 動作後の不感帯（兼、戻り動作待ち時間）

# 感度設定 (キャリブレーション用)
# Mean + (Std * MULTIPLIER) を「動き出し(Trigger)」の閾値とする
TRIGGER_MULTIPLIER = 5.0

# 強打判定の係数
# Trigger閾値のさらに何倍の力が出たら「強打」とするか
# 例: 1.5倍なら、動き出しの1.5倍の力で強打判定
STRONG_RATIO = 1.5

# チャンネル設定 (叩く筋肉)
DOWN_CH = [5, 6]

# ===== グローバル変数・状態管理 =====
rms_buf = [deque(maxlen=30) for _ in range(8)]  # 反応速度優先でWindowサイズ小
ema_val = np.zeros(8)
trigger_thresholds = np.zeros(8) + 999.0  # 初期値は高く設定
strong_threshold = 999.0

# 状態定数
STATE_IDLE = 0  # 待機
STATE_MEASURING = 1  # 計測中（見切り発車中）
STATE_COOLDOWN = 2  # 完了・戻り待ち

current_state = STATE_IDLE
measure_start_time = 0
cooldown_start_time = 0
measure_buffer = []

calibration_done = False
EMA_ALPHA = 0.3  # 平滑化係数 (0.2~0.5推奨)

# ===== シリアル通信 =====
try:
    ser_motor = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.01)
    print(f"[INFO] Arduino connected on {SERIAL_PORT}")
except Exception as e:
    print(f"[WARN] Arduino Connection Failed: {e}")
    ser_motor = None


def send_command(cmd_char):
    """Arduinoへ1文字コマンドを送信"""
    if ser_motor and ser_motor.is_open:
        try:
            ser_motor.write(cmd_char.encode("utf-8"))
        except Exception as e:
            print(f"[ERR] Serial Write: {e}")


# ===== EMG計算 =====
def compute_rms(buf):
    if len(buf) == 0:
        return 0
    return np.sqrt(np.mean(np.array(buf) ** 2))


def calibrate(duration=3.0):
    """
    静止時のノイズを測定し、動作開始(Trigger)と強打(Strong)の閾値を自動決定する
    """
    global trigger_thresholds, strong_threshold, calibration_done

    print(f"\n=== キャリブレーション開始: {duration}秒間、脱力してください ===")
    time.sleep(duration)

    samples = []
    print("... データ解析中 ...")

    # データ収集 (約0.5秒分)
    for _ in range(50):
        # 現在の平滑化された値(EMA)を使用
        samples.append(np.copy(ema_val))
        time.sleep(0.01)

    samples = np.array(samples)

    # 統計計算
    noise_mean = np.mean(samples, axis=0)
    noise_std = np.std(samples, axis=0)

    # 1. 動き出しトリガー閾値 (Mean + N * Std)
    trigger_thresholds = noise_mean + (noise_std * TRIGGER_MULTIPLIER)
    # 安全策: 最低値を保証
    trigger_thresholds = np.maximum(trigger_thresholds, 15.0)

    # 2. 強打判定の閾値 (DOWN_CHのトリガー閾値の平均 × 倍率)
    # 叩く筋肉のトリガー閾値の平均を取得
    base_trigger = np.mean(trigger_thresholds[DOWN_CH])
    strong_threshold = base_trigger * STRONG_RATIO

    calibration_done = True
    print("=== キャリブレーション完了 ===")
    print(f"Trigger Threshold (CH {DOWN_CH}): {np.round(trigger_thresholds[DOWN_CH], 1)}")
    print(f"Strong Threshold: {strong_threshold:.1f} (Trigger x {STRONG_RATIO})")
    print("--------------------------------------------------")


# ===== メインロジック (ハンドラ) =====
def on_emg(emg, movement):
    global current_state, measure_start_time, cooldown_start_time, measure_buffer
    global ema_val

    if emg is None:
        return

    # 1. RMS & EMA計算
    for ch in range(8):
        rms_buf[ch].append(emg[ch])
        rms = compute_rms(rms_buf[ch])
        ema_val[ch] = EMA_ALPHA * rms + (1 - EMA_ALPHA) * ema_val[ch]

    if not calibration_done:
        return

    # 叩く動作の強度（対象チャンネルの最大値）
    down_intensity = np.max([ema_val[ch] for ch in DOWN_CH])
    now = time.time()

    # --- ステートマシン制御 ---

    # [状態1: IDLE] トリガー待ち
    if current_state == STATE_IDLE:
        # DOWNチャンネルのいずれかが閾値を超えたか？
        is_triggered = False
        for ch in DOWN_CH:
            if ema_val[ch] > trigger_thresholds[ch]:
                is_triggered = True
                break

        if is_triggered:
            # ★見切り発車 (Initial Dash)
            # とりあえず中速で動き出す
            send_command("I")

            current_state = STATE_MEASURING
            measure_start_time = now
            measure_buffer = []

    # [状態2: MEASURING] 計測中 (50ms間)
    elif current_state == STATE_MEASURING:
        measure_buffer.append(down_intensity)

        # 時間経過チェック
        if (now - measure_start_time) * 1000 >= MEASURE_DURATION_MS:
            # 平均RMSを計算して最終判定
            if len(measure_buffer) > 0:
                avg_val = np.mean(measure_buffer)

                if avg_val > strong_threshold:
                    # 強打 (Strong) -> 加速
                    send_command("S")
                    print(f"[HIT] STRONG (Val:{avg_val:.1f})")
                else:
                    # 弱打 (Weak) -> 減速/ブレーキ
                    send_command("W")
                    print(f"[HIT] weak   (Val:{avg_val:.1f})")

            current_state = STATE_COOLDOWN
            cooldown_start_time = now

    # [状態3: COOLDOWN] 戻り動作待ち
    elif current_state == STATE_COOLDOWN:
        # 指定時間経過後、筋肉が脱力していれば戻る
        if (now - cooldown_start_time) * 1000 >= COOLDOWN_MS:

            is_quiet = True
            for ch in DOWN_CH:
                # チャタリング防止: トリガー閾値を少し下回るまで待つ
                if ema_val[ch] > trigger_thresholds[ch]:
                    is_quiet = False
                    break

            if is_quiet:
                # 初期位置へ戻る (Return)
                send_command("R")
                current_state = STATE_IDLE


# ===== メイン処理 =====
def main():
    print("Myo Armband 接続待機中...")
    m = Myo(mode=emg_mode.RAW)
    m.connect()

    # データ処理は別スレッドで行う（安定性のため）
    m.add_emg_handler(on_emg)

    def worker():
        try:
            while True:
                m.run()
        except:
            pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # 初期化動作
    m.set_leds([0, 255, 0], [0, 255, 0])
    m.vibrate(1)
    send_command("R")  # サーボを初期位置へ

    # キャリブレーション
    try:
        calibrate()
    except KeyboardInterrupt:
        pass

    print("\n==== DRUM MODE START ====")
    print("Commands: 'I'(Init) -> 'S'(Strong)/'W'(Weak) -> 'R'(Return)")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n終了処理中...")
        m.disconnect()
        if ser_motor:
            ser_motor.close()
        print("完了")


if __name__ == "__main__":
    main()