import time
import numpy as np
from collections import deque
from pyomyo import Myo, emg_mode
import serial
import threading
import csv
import datetime
import queue  # ★追加: スレッド間通信用

# ===== Arduino設定 =====
SERIAL_MOTOR = "COM5"
SERIAL_SENSOR = "COM6"
BAUDRATE = 115200

# ... (接続処理は変更なし) ...
try:
    ser_motor = serial.Serial(SERIAL_MOTOR, BAUDRATE, timeout=0.01)
    print(f"Motor Arduino connected: {SERIAL_MOTOR}")
except:
    ser_motor = None

try:
    ser_sensor = serial.Serial(SERIAL_SENSOR, BAUDRATE, timeout=0.01)
    print(f"Sensor Arduino connected: {SERIAL_SENSOR}")
except:
    ser_sensor = None

# ===== 制御パラメータ =====
UP_CH = [1, 2]
DOWN_CH = [5, 6]
MEASURE_DURATION_MS = 60
COOLDOWN_MS = 150
# ★調整済みパラメータ
RISE_SENSITIVITY = 4.0  # 6.0 -> 4.0 に下げて感度アップ
UP_HOLD_SENSITIVITY = 5.0
STRONG_RATIO = 1.2
EMA_ALPHA = 0.3  # 0.2 -> 0.3 に上げて反応速度アップ
CALIB_DURATION = 3.0

# ===== グローバル変数 =====
rms_buf = [deque(maxlen=30) for _ in range(8)]
ema_val = np.zeros(8)
prev_ema_val = np.zeros(8)

# ★変更: CSV関連は直接持たず、Queueを使う
log_queue = queue.Queue()
is_running = True  # 全スレッド停止用フラグ

# 閾値格納用 (変更なし)
rise_thresholds = np.zeros(8) + 999.0
level_thresholds = np.zeros(8) + 999.0
strong_threshold = 999.0
calibration_done = False

# 状態管理 (変更なし)
STATE_IDLE = 0
STATE_ATTACKING = 1
STATE_COOLDOWN = 2
current_state = STATE_IDLE
measure_start_time = 0
cooldown_start_time = 0
measure_buffer = []
is_holding_up = False

# センサーデータ格納用
latest_sensor_data = [0.0, 0.0, 0.0]
sensor_lock = threading.Lock()  # ★追加: データ共有の排他制御用


# ===== ★追加: ロギング専用スレッド =====
def logging_loop(filename):
    """
    Queueに溜まったデータをひたすらCSVに書き込む裏方さん。
    モーター制御を邪魔しないために別スレッドで動かす。
    """
    try:
        with open(filename, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            header = ["timestamp"] + [f"ch{i+1}" for i in range(8)] + ["ard_micros", "vib1_z", "vib2_z", "state"]
            csv_writer.writerow(header)
            print(f"CSV Recording started: {filename}")

            while is_running or not log_queue.empty():
                try:
                    # データが来るのを待つ (timeout付きにして終了判定できるようにする)
                    row = log_queue.get(timeout=0.1)
                    csv_writer.writerow(row)
                    log_queue.task_done()
                except queue.Empty:
                    continue
    except Exception as e:
        print(f"CSV Logging Error: {e}")


# ===== センサー読み取りスレッド =====
def sensor_read_loop():
    global latest_sensor_data
    while is_running:
        if ser_sensor and ser_sensor.is_open:
            try:
                if ser_sensor.in_waiting > 0:  # データがあるときだけ読む
                    line = ser_sensor.readline().decode("utf-8").strip()
                    if line:
                        parts = line.split(",")
                        if len(parts) >= 3:
                            # ★Lockを使って書き込み中の読み出しを防ぐ
                            with sensor_lock:
                                latest_sensor_data = [float(parts[0]), float(parts[1]), float(parts[2])]
                else:
                    time.sleep(0.001)  # CPU使用率を下げるため
            except:
                pass
        else:
            time.sleep(0.1)


# ===== 関数 (変更なし部分は省略) =====
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


def calibrate(duration=CALIB_DURATION):
    # (ロジックは元のままなので省略、ただしCSV保存はQueue経由になるため影響なし)
    global rise_thresholds, level_thresholds, strong_threshold, calibration_done
    print("\n=== キャリブレーション: 3秒間 脱力してください ===")
    time.sleep(duration)
    val_samples = []
    diff_samples = []
    print("... ノイズ学習中 ...")
    for _ in range(10):
        time.sleep(0.01)
    for _ in range(50):
        current_val = np.copy(ema_val)
        current_diff = np.abs(ema_val - prev_ema_val)
        val_samples.append(current_val)
        diff_samples.append(current_diff)
        time.sleep(0.01)

    val_mean = np.mean(val_samples, axis=0)
    val_std = np.std(val_samples, axis=0)
    level_thresholds = val_mean + (val_std * UP_HOLD_SENSITIVITY)
    level_thresholds = np.maximum(level_thresholds, 15.0)

    diff_mean = np.mean(diff_samples, axis=0)
    diff_std = np.std(diff_samples, axis=0)
    rise_thresholds = diff_mean + (diff_std * RISE_SENSITIVITY)
    rise_thresholds = np.maximum(rise_thresholds, 2.0)

    strong_threshold = np.mean(level_thresholds[DOWN_CH]) * STRONG_RATIO
    calibration_done = True
    print("=== 完了 ===")


# ===== メインコールバック =====
def on_emg(emg, movement):
    global current_state, measure_start_time, cooldown_start_time, measure_buffer
    global is_holding_up, ema_val, prev_ema_val

    # ★処理開始時間を記録 (これをCSVのタイムスタンプにする)
    now = time.time()

    if emg is None:
        return

    prev_ema_val = np.copy(ema_val)
    for ch in range(8):
        rms_buf[ch].append(emg[ch])
        rms = compute_rms(rms_buf[ch])
        ema_val[ch] = EMA_ALPHA * rms + (1 - EMA_ALPHA) * ema_val[ch]

    # ★変更: ここでCSV書き込みを待たない！Queueに入れるだけ。
    # 最新のセンサーデータを安全に取得
    with sensor_lock:
        current_sensor_data = list(latest_sensor_data)

    # Queueにデータを放り込む (タイムスタンプ, EMG, センサーデータ, 現在の状態)
    # 状態(current_state)も記録しておくと後で解析しやすいです
    log_data = [now] + ema_val.tolist() + current_sensor_data + [current_state]
    log_queue.put(log_data)

    if not calibration_done:
        return

    # --- 以下、制御ロジック (元のまま) ---
    up_level = np.max([ema_val[ch] for ch in UP_CH])
    down_slope_list = [(ema_val[ch] - prev_ema_val[ch]) for ch in DOWN_CH]
    max_down_slope = np.max(down_slope_list)
    down_level = np.max([ema_val[ch] for ch in DOWN_CH])

    if current_state == STATE_IDLE:
        is_attack_triggered = False
        for i, ch in enumerate(DOWN_CH):
            if down_slope_list[i] > rise_thresholds[ch]:
                is_attack_triggered = True
                break

        if is_attack_triggered:
            send_cmd("I")
            current_state = STATE_ATTACKING
            measure_start_time = now
            measure_buffer = []
            is_holding_up = False
            print(f">> ATTACK! (Slope: {max_down_slope:.2f})")
            return

        # ★デバッグ用: 攻撃用の変化量が閾値に対してどうなっているか見る
        if max_down_slope > 1.0:  # 少しでも反応があったら表示
            # どのチャンネルが反応しているか知りたいので詳細表示
            ch_idx = np.argmax(down_slope_list)
            target_ch = DOWN_CH[ch_idx]
            print(f"[DEBUG] Slope:{max_down_slope:.2f} / Thr:{rise_thresholds[target_ch]:.2f} (ch{target_ch+1})")

        is_up_active = up_level > np.mean(level_thresholds[UP_CH])
        if is_up_active and not is_holding_up:
            send_cmd("L")
            is_holding_up = True
        elif not is_up_active and is_holding_up:
            send_cmd("R")
            is_holding_up = False

    elif current_state == STATE_ATTACKING:
        measure_buffer.append(down_level)
        if (now - measure_start_time) * 1000 >= MEASURE_DURATION_MS:
            if len(measure_buffer) > 0:
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
    global is_running
    print("Myo接続中...")
    m = Myo(mode=emg_mode.RAW)
    m.connect()
    m.add_emg_handler(on_emg)

    # 1. Myoスレッド
    t_myo = threading.Thread(target=lambda: m.run(), daemon=True)
    t_myo.start()

    # 2. センサースレッド
    if ser_sensor:
        t_sensor = threading.Thread(target=sensor_read_loop, daemon=True)
        t_sensor.start()

    m.set_leds([0, 255, 255], [0, 255, 255])
    m.vibrate(1)
    send_cmd("R")

    import os

    # 3. ★ロギングスレッドの開始
    # フォルダ1に保存するように指定
    save_dir = r"c:\Users\hrsyn\Desktop\gitPython\1"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = os.path.join(save_dir, datetime.datetime.now().strftime("emg_data_%Y%m%d_%H%M%S.csv"))
    t_logger = threading.Thread(target=logging_loop, args=(filename,), daemon=True)
    t_logger.start()

    calibrate()

    print("\n==== 立ち上がり検出モード (V19) ====")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        is_running = False  # ループを停止させる
        m.disconnect()
        if ser_motor:
            ser_motor.close()
        if ser_sensor:
            ser_sensor.close()

        # ロギングスレッドの終了待ち
        print("Saving remaining data...")
        t_logger.join(timeout=2.0)
        print("Done.")


if __name__ == "__main__":
    main()
