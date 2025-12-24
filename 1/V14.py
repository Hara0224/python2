import time
import numpy as np
from collections import deque
from pyomyo import Myo, emg_mode
import serial
import threading
import sys

# ===== 設定項目 (環境に合わせて変更してください) =====
# ArduinoのCOMポート (デバイスマネージャーで確認)
SERIAL_PORT = "COM5"
BAUDRATE = 115200

# 判定パラメータ
# 1145BWは高速なため、判定時間を短く(50ms)設定しています
MEASURE_DURATION_MS = 50
COOLDOWN_MS = 150  # 動作完了後の不感帯

# 閾値設定 (キャリブレーションで決定されますが、初期値を設定)
# これを超えたら強打(S)、超えなければ弱打(W)と判定
STRONG_THRESHOLD_RMS = 60.0

# トリガー感度 (ノイズフロアの何倍で反応するか)
TRIGGER_SENSITIVITY = 5.0

# チャンネル設定
# 叩く動作に使う筋肉のチャンネル
DOWN_CH = [5, 6]

# ===== Arduino通信セットアップ =====
try:
    ser_motor = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.01)
    print(f"Arduino connected on {SERIAL_PORT}")
except Exception as e:
    print(f"Error opening serial: {e}")
    print("Arduinoなしで動作確認モードに入ります...")
    ser_motor = None

# ===== グローバル変数 =====
rms_buf = [deque(maxlen=30) for _ in range(8)]  # 反応速度優先でWindowサイズ30
current_rms_values = np.zeros(8)
trigger_thresholds = np.zeros(8) + 999.0  # 初期値は反応しないように高く
calibration_done = False

# 状態管理
STATE_IDLE = 0
STATE_MEASURING = 1
STATE_COOLDOWN = 2

current_state = STATE_IDLE
measure_start_time = 0
cooldown_start_time = 0
measure_buffer = []


# ===== 関数群 =====
def compute_rms(buf):
    """二乗平均平方根(RMS)を計算"""
    if len(buf) == 0:
        return 0
    arr = np.array(buf)
    return np.sqrt(np.mean(arr**2))


def send_command(cmd_char):
    """Arduinoへ1文字コマンドを送信"""
    if ser_motor and ser_motor.is_open:
        ser_motor.write(cmd_char.encode("utf-8"))


def calibrate(duration=3.0):
    """ノイズフロアを測定してトリガー閾値を決定"""
    global trigger_thresholds, calibration_done
    print(f"\n=== キャリブレーション開始: {duration}秒間リラックスしてください ===")
    time.sleep(duration)

    samples = []
    print("... データ収集中 ...")
    # 約0.5秒分のデータを取得
    for _ in range(50):
        samples.append(np.copy(current_rms_values))
        time.sleep(0.01)

    samples = np.array(samples)
    noise_mean = np.mean(samples, axis=0)
    noise_std = np.std(samples, axis=0)

    # 平均 + (標準偏差 × 感度) をトリガー閾値とする
    trigger_thresholds = noise_mean + (noise_std * TRIGGER_SENSITIVITY)

    # 安全のため、低すぎる閾値を防ぐ (誤作動防止)
    trigger_thresholds = np.maximum(trigger_thresholds, 15.0)

    calibration_done = True
    print("=== キャリブレーション完了 ===")
    print(f"Trigger Thresholds (CH {DOWN_CH}): {np.round(trigger_thresholds[DOWN_CH], 1)}")
    print(f"Strong Impact Threshold: {STRONG_THRESHOLD_RMS}")


# ===== EMGハンドラ (メインロジック) =====
def on_emg(emg, movement):
    global current_state, measure_start_time, cooldown_start_time, measure_buffer

    if emg is None:
        return

    # 1. RMS計算
    for ch in range(8):
        rms_buf[ch].append(emg[ch])
        current_rms_values[ch] = compute_rms(rms_buf[ch])

    if not calibration_done:
        return

    # 2. トリガー判定用の値 (叩く筋肉の最大値)
    down_intensity = np.max([current_rms_values[ch] for ch in DOWN_CH])

    now = time.time()

    # --- ステートマシン ---

    # [状態: 待機] トリガー待ち
    if current_state == STATE_IDLE:
        # 閾値を超えたかチェック
        is_triggered = False
        for ch in DOWN_CH:
            if current_rms_values[ch] > trigger_thresholds[ch]:
                is_triggered = True
                break

        if is_triggered:
            # ★見切り発車: とりあえず中速で動き出し ('I')
            send_command("I")
            # print(">> DASH (Initial)")

            # 計測開始
            current_state = STATE_MEASURING
            measure_start_time = now
            measure_buffer = []

    # [状態: 計測中] 動きながら強弱を判断
    elif current_state == STATE_MEASURING:
        measure_buffer.append(down_intensity)

        # 指定時間 (50ms) 経過後に判定
        if (now - measure_start_time) * 1000 >= MEASURE_DURATION_MS:
            if len(measure_buffer) > 0:
                avg_rms = np.mean(measure_buffer)

                # 強弱判定
                if avg_rms > STRONG_THRESHOLD_RMS:
                    # 強打: ブーストコマンド ('S')
                    send_command("S")
                    print(f"[HIT] STRONG! (RMS:{avg_rms:.1f})")
                else:
                    # 弱打: ブレーキコマンド ('W')
                    send_command("W")
                    print(f"[HIT] weak... (RMS:{avg_rms:.1f})")

            # クールダウンへ
            current_state = STATE_COOLDOWN
            cooldown_start_time = now

    # [状態: クールダウン] 完了待ち -> 戻り動作
    elif current_state == STATE_COOLDOWN:
        # 指定時間が経過したら戻る
        if (now - cooldown_start_time) * 1000 >= COOLDOWN_MS:

            # 筋肉が脱力しているかチェック（チャタリング防止）
            all_quiet = True
            for ch in DOWN_CH:
                # 閾値より少し低い値に戻るまで待つ
                if current_rms_values[ch] > trigger_thresholds[ch]:
                    all_quiet = False
                    break

            if all_quiet:
                # 初期位置へ戻るコマンド ('R')
                send_command("R")
                current_state = STATE_IDLE


# ===== メイン処理 =====
def main():
    print("Myo Armband 接続中...")
    try:
        m = Myo(mode=emg_mode.RAW)
        m.connect()
    except Exception as e:
        print(f"Myo接続エラー: {e}")
        return

    m.add_emg_handler(on_emg)

    # Myoからのデータ受信を別スレッドで実行
    def worker():
        try:
            while True:
                m.run()
        except:
            pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # LED通知 & 振動
    m.set_leds([0, 0, 255], [0, 0, 255])
    m.vibrate(1)

    # 起動時に一度リセット位置へ
    send_command("R")

    # キャリブレーション実行
    calibrate()

    print("\n=== システム準備完了 ===")
    print(f"判定時間: {MEASURE_DURATION_MS}ms")
    print(f"強打判定閾値: {STRONG_THRESHOLD_RMS}")
    print("Ctrl+C で終了します")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n終了処理中...")
        m.disconnect()
        if ser_motor:
            ser_motor.close()
        print("Bye.")


if __name__ == "__main__":
    main()
