import time
import numpy as np
from collections import deque
from pyomyo import Myo, emg_mode
import serial
import threading
import csv
import datetime
import os

# ==========================================
#  V20: Hysteresis Control Version
#  特徴: UP判定に「粘り」を持たせ、チャタリングを防止
# ==========================================

# ===== Arduino設定 =====
SERIAL_MOTOR = "COM4"
SERIAL_SENSOR = "COM5"  # ログに合わせて修正
BAUDRATE = 115200

# ... (接続処理) ...
try:
    ser_motor = serial.Serial(SERIAL_MOTOR, BAUDRATE, timeout=1.0)
    print(f"Motor Arduino connected: {SERIAL_MOTOR}")
    print("Waiting for Motor Arduino restart (2s)...")
    time.sleep(2.0)
except:
    ser_motor = None
    print(f"!! Motor Arduino NOT connected !!")

try:
    ser_sensor = serial.Serial(SERIAL_SENSOR, BAUDRATE, timeout=0.01)
    print(f"Sensor Arduino connected: {SERIAL_SENSOR}")
except:
    ser_sensor = None
    print(f"!! Sensor Arduino NOT connected !!")

# ===== 制御パラメータ =====
UP_CH = [1, 2]   # 撓屈（振りかぶり）
DOWN_CH = [6]    # 尺屈（振り下ろし）

# ★ヒステリシス閾値 (ここを調整)
UP_ON_THRESHOLD  = 7.5  # Zスコアがこれを超えたら「UP」
UP_OFF_THRESHOLD = 4.0  # Zスコアがこれを下回ったら「Release」

MEASURE_DURATION_MS = 150
COOLDOWN_MS = 500
STRONG_RATIO = 13.0
EMA_ALPHA = 0.2
K_SIGMA = 5.7   # Attack(Down)の感度
CALIB_DURATION = 5.0

# ===== グローバル変数 =====
rms_buf = [deque(maxlen=30) for _ in range(8)]
ema_val = np.zeros(8)
prev_ema_val = np.zeros(8)

csv_file = None
csv_writer = None
is_running = True

mean = np.zeros(8)
std = np.ones(8)
strong_threshold = 999.0
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
    if len(buf) == 0: return 0
    return np.sqrt(np.mean(np.array(buf) ** 2))

def send_cmd(cmd):
    if ser_motor and ser_motor.is_open:
        try:
            # print(f"  [CMD] Sending: {cmd}") # ログがうるさい場合はコメントアウト
            ser_motor.write(cmd.encode("utf-8"))
            ser_motor.flush()
        except Exception as e:
            print(f"  [CMD] Error: {e}")

def calibrate(duration=CALIB_DURATION):
    global mean, std, calibration_done, strong_threshold
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
            
    # Attack強度の自動設定
    print("\n【設定】強弱の閾値を決定します。")
    # (ここは簡易版として、計算だけで出すなら以下のように設定可能)
    # strong_threshold = mean[DOWN_CH[0]] + std[DOWN_CH[0]] * STRONG_RATIO
    
    # 以前のように実測するならここで入力待ちを入れるが、今回は省略して計算値にする場合:
    strong_thr_list = []
    for ch in DOWN_CH:
        strong_thr_list.append(mean[ch] + std[ch] * STRONG_RATIO)
    strong_threshold = np.mean(strong_thr_list)

    calibration_done = True
    print("=== 完了 ===")
    print(f"DEBUG: Mean: {mean}")
    print(f"DEBUG: Strong Thr: {strong_threshold:.2f}")

# ===== メインコールバック =====
def on_emg(emg, movement):
    global current_state, measure_start_time, cooldown_start_time, measure_buffer
    global is_holding_up, ema_val, prev_ema_val

    now = time.time()
    if emg is None: return

    # 動作確認ログ
    log_counter = getattr(on_emg, "counter", 0) + 1
    on_emg.counter = log_counter
    if log_counter % 100 == 0:
        ch5_val = ema_val[5]
        ch6_val = ema_val[6]
        print(f"[Status] CH5:{ch5_val:.1f} CH6:{ch6_val:.1f}")

    prev_ema_val = np.copy(ema_val)
    for ch in range(8):
        rms_buf[ch].append(emg[ch])
        rms = compute_rms(rms_buf[ch])
        ema_val[ch] = EMA_ALPHA * rms + (1 - EMA_ALPHA) * ema_val[ch]

    # CSV書き込み
    if csv_writer:
        try:
            log_data = [now] + ema_val.tolist() + latest_sensor_data + [current_state]
            csv_writer.writerow(log_data)
        except:
            pass

    if not calibration_done: return

    # Z-score計算
    z_scores = (ema_val - mean) / (std + 1e-6)

    # 判定用スコア
    down_z_list = [z_scores[ch] for ch in DOWN_CH]
    max_down_z = np.max(down_z_list)
    
    up_z_list = [z_scores[ch] for ch in UP_CH]
    max_up_z = np.max(up_z_list)

    down_level = np.max([ema_val[ch] for ch in DOWN_CH])

    # --- ステートマシン ---
    if current_state == STATE_IDLE:
        
        # 1. Attack Trigger (最優先)
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
            print(f">> ATTACK! (Z-Score: {max_down_z:.2f})")
            return

        # 2. UP Trigger (ヒステリシス制御)
        # -----------------------------------------------
        # 上げる時: UP_ON_THRESHOLD (7.5) を超える必要がある
        # 下げる時: UP_OFF_THRESHOLD (4.0) を下回る必要がある
        # -----------------------------------------------
        if is_holding_up:
            # 今「上げている」なら、下回るまで我慢
            if max_up_z < UP_OFF_THRESHOLD:
                send_cmd("R")
                is_holding_up = False
                print(f">> UP Released. (Z: {max_up_z:.2f})")
        else:
            # 今「下がっている」なら、超えるまで我慢
            if max_up_z > UP_ON_THRESHOLD:
                send_cmd("L")
                is_holding_up = True
                print(f">> UP! (Z: {max_up_z:.2f}) - LOCKED")

        # Debug (Zスコアが高いときだけ表示)
        if max_up_z > 4.0 and log_counter % 20 == 0:
            status = "HOLD" if is_holding_up else "OFF"
            print(f"[DEBUG UP] Z:{max_up_z:.2f} State:{status}")

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
    global is_running, csv_file, csv_writer
    print("Myo接続中...")
    m = Myo(mode=emg_mode.RAW)
    m.connect()
    m.add_emg_handler(on_emg)

    # スレッド開始
    t_myo = threading.Thread(target=m.run, daemon=True)
    t_myo.start()
    
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

    # CSV設定
    save_dir = r"C:\Users\hrsyn\Desktop\masterPY\1"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    filename = os.path.join(save_dir, datetime.datetime.now().strftime("emg_data_%Y%m%d_%H%M%S.csv"))
    
    try:
        csv_file = open(filename, "w", newline="")
        csv_writer = csv.writer(csv_file)
        header = ["timestamp"] + [f"ch{i+1}" for i in range(8)] + ["ard_micros", "vib1_z", "vib2_z", "state"]
        csv_writer.writerow(header)
        print(f"CSV Recording started: {filename}")
    except Exception as e:
        print(f"Failed to open CSV: {e}")

    calibrate()

    print("\n==== 立ち上がり検出モード (V20: Hysteresis) ====")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        is_running = False
        m.disconnect()
        if ser_motor: ser_motor.close()
        if ser_sensor: ser_sensor.close()
        if csv_file: csv_file.close()
        os._exit(0)

if __name__ == "__main__":
    main()