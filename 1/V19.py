import time
import numpy as np
from collections import deque
from pyomyo import Myo, emg_mode
import serial
import threading
import csv
import datetime
import os

# ===== Arduino設定 =====
SERIAL_MOTOR = "COM4"
SERIAL_SENSOR = "COM5"
BAUDRATE = 115200

# ... (接続処理) ...
try:
    # timeout: 0.01 -> 1.0 (motor_check.pyに合わせる)
    ser_motor = serial.Serial(SERIAL_MOTOR, BAUDRATE, timeout=1.0)
    print(f"Motor Arduino connected: {SERIAL_MOTOR}")
    print("Waiting for Motor Arduino restart (2s)...")
    time.sleep(2.0)
except:
    ser_motor = None

try:
    ser_sensor = serial.Serial(SERIAL_SENSOR, BAUDRATE, timeout=0.01)
    print(f"Sensor Arduino connected: {SERIAL_SENSOR}")
except:
    ser_sensor = None

# ===== 制御パラメータ =====
UP_CH = [1, 2]
DOWN_CH = [5]  # 必要に応じて [5, 6] に変更
MEASURE_DURATION_MS = 150
COOLDOWN_MS = 500
EMA_ALPHA = 0.3  # 反応速度
K_SIGMA = 5.7  # トリガー感度
CALIB_DURATION = 3.0

# ★安全装置付きキャリブレーションの基準範囲
SAFE_WEAK_MIN = 60.0
SAFE_WEAK_MAX = 70.0
SAFE_STRONG_MIN = 80.0
SAFE_STRONG_MAX = 100.0

# 実際の判定に使われる閾値（キャリブレーションで更新）
weak_threshold = SAFE_WEAK_MIN  # デフォルト 60.0
strong_threshold = SAFE_STRONG_MIN  # デフォルト 80.0

# ===== グローバル変数 =====
rms_buf = [deque(maxlen=30) for _ in range(8)]
ema_val = np.zeros(8)
prev_ema_val = np.zeros(8)

csv_file = None
csv_writer = None
is_running = True

mean = np.zeros(8)
std = np.ones(8)
calibration_done = False

STATE_IDLE = 0
STATE_ATTACKING = 1
STATE_COOLDOWN = 2
current_state = STATE_IDLE
measure_start_time = 0
cooldown_start_time = 0
measure_buffer = []
is_holding_up = False

latest_sensor_data = [0.0, 0.0, 0.0]
sensor_lock = threading.Lock()


# ===== センサー読み取りスレッド =====
def sensor_read_loop():
    global latest_sensor_data
    while is_running:
        if ser_sensor and ser_sensor.is_open:
            try:
                if ser_sensor.in_waiting > 0:
                    line = ser_sensor.readline().decode("utf-8").strip()
                    if line:
                        parts = line.split(",")
                        if len(parts) >= 3:
                            with sensor_lock:
                                latest_sensor_data = [float(parts[0]), float(parts[1]), float(parts[2])]
                else:
                    time.sleep(0.001)
            except:
                pass
        else:
            time.sleep(0.1)


# ===== 関数 =====
def compute_rms(buf):
    if len(buf) == 0:
        return 0
    return np.sqrt(np.mean(np.array(buf) ** 2))


def send_cmd(cmd):
    if ser_motor and ser_motor.is_open:
        try:
            # print(f"  [CMD] Sending: {cmd}")
            ser_motor.write(cmd.encode("utf-8"))
            ser_motor.flush()
        except Exception as e:
            print(f"  [CMD] Error: {e}")


def calibrate(duration=CALIB_DURATION):
    global mean, std, calibration_done
    global weak_threshold, strong_threshold

    print("\n=== キャリブレーション: 3秒間 脱力してください ===")

    cal_cols = [[] for _ in range(8)]
    start_time = time.time()
    while time.time() - start_time < duration:
        time.sleep(0.01)
        for ch in range(8):
            cal_cols[ch].append(ema_val[ch])

    for ch in range(8):
        arr = np.array(cal_cols[ch])
        mean[ch] = np.mean(arr)
        std[ch] = np.std(arr, ddof=1)
        if std[ch] < 1e-6 or np.isnan(std[ch]):
            std[ch] = 1e-6

    print("\n----------------------------------------")
    print("【設定】強弱の閾値を決定します（安全装置付き）。")

    # --- 1. 弱い動作 (Weak) ---
    print(f"3秒後に『弱い』動作を3秒間続けてください (目標: {SAFE_WEAK_MIN}-{SAFE_WEAK_MAX})...")
    time.sleep(3.0)
    print(">> 計測中... (Weak)")

    weak_vals = []
    start_time = time.time()
    while time.time() - start_time < 3.0:
        time.sleep(0.01)
        val = np.max([ema_val[ch] for ch in DOWN_CH])
        weak_vals.append(val)

    measured_weak_max = np.max(weak_vals) if weak_vals else 0.0
    print(f">> 計測値(Max): {measured_weak_max:.2f}")

    # ★Weak判定ロジック (範囲外ならデフォルト値)
    if SAFE_WEAK_MIN <= measured_weak_max <= SAFE_WEAK_MAX:
        weak_threshold = measured_weak_max
        print(f"  -> 範囲内採用: {weak_threshold:.2f}")
    else:
        weak_threshold = SAFE_WEAK_MIN  # デフォルト60
        print(f"  -> 範囲外のためデフォルト採用: {weak_threshold:.2f}")

    # --- 2. 強い動作 (Strong) ---
    print(f"\n続いて、3秒後に『強い』動作を3秒間続けてください (目標: {SAFE_STRONG_MIN}-{SAFE_STRONG_MAX})...")
    time.sleep(3.0)
    print(">> 計測中... (Strong)")

    strong_vals = []
    start_time = time.time()
    while time.time() - start_time < 3.0:
        time.sleep(0.01)
        val = np.max([ema_val[ch] for ch in DOWN_CH])
        strong_vals.append(val)

    measured_strong_max = np.max(strong_vals) if strong_vals else 0.0
    print(f">> 計測値(Max): {measured_strong_max:.2f}")

    # ★Strong判定ロジック (範囲外ならデフォルト値)
    if SAFE_STRONG_MIN <= measured_strong_max <= SAFE_STRONG_MAX:
        strong_threshold = measured_strong_max
        print(f"  -> 範囲内採用: {strong_threshold:.2f}")
    else:
        strong_threshold = SAFE_STRONG_MIN  # デフォルト80
        print(f"  -> 範囲外のためデフォルト採用: {strong_threshold:.2f}")

    calibration_done = True
    print("\n=== キャリブレーション完了 ===")
    print(f"★確定閾値 :: Weak: {weak_threshold:.1f} / Strong: {strong_threshold:.1f}")
    print(f"  (これより {strong_threshold:.1f} 以上なら Strong判定)")


# ===== メインコールバック =====
def on_emg(emg, movement):
    global current_state, measure_start_time, cooldown_start_time, measure_buffer
    global is_holding_up, ema_val, prev_ema_val

    now = time.time()
    if emg is None:
        return

    # ログ表示更新頻度
    log_counter = getattr(on_emg, "counter", 0) + 1
    on_emg.counter = log_counter

    prev_ema_val = np.copy(ema_val)
    for ch in range(8):
        rms_buf[ch].append(emg[ch])
        rms = compute_rms(rms_buf[ch])
        ema_val[ch] = EMA_ALPHA * rms + (1 - EMA_ALPHA) * ema_val[ch]

    if csv_writer:
        try:
            log_data = [now] + ema_val.tolist() + latest_sensor_data + [current_state]
            csv_writer.writerow(log_data)
        except Exception:
            pass

    if not calibration_done:
        return

    # --- 制御ロジック ---
    z_scores = (ema_val - mean) / (std + 1e-6)

    # Attack (DOWN) の判定
    down_z_list = [z_scores[ch] for ch in DOWN_CH]
    max_down_z = np.max(down_z_list)

    # UP (Lift) の判定
    up_z_list = [z_scores[ch] for ch in UP_CH]
    max_up_z = np.max(up_z_list)

    # 生の値（強弱判定用）
    down_level = np.max([ema_val[ch] for ch in DOWN_CH])

    if current_state == STATE_IDLE:
        # Attack Trigger
        is_attack_triggered = False
        for ch in DOWN_CH:
            if z_scores[ch] > K_SIGMA:
                is_attack_triggered = True
                break

        if is_attack_triggered:
            send_cmd("I")
            current_state = STATE_ATTACKING
            measure_start_time = now
            measure_buffer = []
            is_holding_up = False
            print(f">> ATTACK! (Z: {max_down_z:.2f})")
            return

        # UP Trigger
        is_up_active = False
        for ch in UP_CH:
            if z_scores[ch] > K_SIGMA:
                is_up_active = True
                break

        if is_up_active and not is_holding_up:
            send_cmd("L")
            is_holding_up = True
            print(f">> UP! (Z: {max_up_z:.2f})")
        elif not is_up_active and is_holding_up:
            send_cmd("R")
            is_holding_up = False
            print(">> UP Released.")

    elif current_state == STATE_ATTACKING:
        measure_buffer.append(down_level)
        # 計測時間経過後
        if (now - measure_start_time) * 1000 >= MEASURE_DURATION_MS:
            if len(measure_buffer) > 0:
                peak_val = np.max(measure_buffer)

                # ★修正: 確定した strong_threshold を使ってシンプルに分岐
                if peak_val >= strong_threshold:
                    # Strong
                    print(f"[HIT] STRONG !!! (Level:{peak_val:.1f} >= {strong_threshold:.1f})")
                    send_cmd("S")
                else:
                    # Weak (Strong未満はすべてWeak扱い)
                    print(f"[HIT] weak       (Level:{peak_val:.1f} < {strong_threshold:.1f})")
                    send_cmd("W")

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
    print(">> Myo.connect() finished.")
    m.add_emg_handler(on_emg)

    # 1. Myoスレッド
    def myo_worker():
        while is_running:
            m.run()

    t_myo = threading.Thread(target=myo_worker, daemon=True)
    t_myo.start()

    # 2. センサースレッド
    if ser_sensor:
        t_sensor = threading.Thread(target=sensor_read_loop, daemon=True)
        t_sensor.start()

    print(">> Reset motor...")
    send_cmd("R")
    time.sleep(1.0)
    print(">> Motor Test: LIFT (L)")
    send_cmd("L")
    time.sleep(1.0)
    print(">> Motor Test: Release (R)")
    send_cmd("R")

    print(">> Calling calibrate()...")

    # 3. CSV設定
    save_dir = r"C:\Users\hrsyn\Desktop\masterPY\1"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, datetime.datetime.now().strftime("emg_data_%Y%m%d_%H%M%S.csv"))

    global csv_file, csv_writer
    try:
        csv_file = open(filename, "w", newline="")
        csv_writer = csv.writer(csv_file)
        header = ["timestamp"] + [f"ch{i+1}" for i in range(8)] + ["ard_micros", "vib1_z", "vib2_z", "state"]
        csv_writer.writerow(header)
        print(f"CSV Recording started: {filename}")
    except Exception as e:
        print(f"Failed to open CSV: {e}")

    calibrate()

    print("\n==== 立ち上がり検出モード (Safeguard) ====")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        is_running = False
        m.disconnect()
        if ser_motor:
            ser_motor.close()
        if ser_sensor:
            ser_sensor.close()
        if csv_file:
            csv_file.close()
        os._exit(0)


if __name__ == "__main__":
    main()
