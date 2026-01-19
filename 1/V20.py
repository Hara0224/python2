import time
import numpy as np
from collections import deque
from pyomyo import Myo, emg_mode
import serial
import threading
import csv
import datetime
import os
import sys

# ===== Arduino設定 =====
SERIAL_MOTOR = "COM4"
SERIAL_SENSOR = "COM5"
BAUDRATE = 115200

ser_motor = None
ser_sensor = None

try:
    ser_motor = serial.Serial(SERIAL_MOTOR, BAUDRATE, timeout=0.1)
    print(f"Motor Arduino connected: {SERIAL_MOTOR}")
    time.sleep(2.0)
    ser_motor.reset_input_buffer()
    ser_motor.reset_output_buffer()
except Exception as e:
    print(f"Motor Connection Failed: {e}")
    ser_motor = None

try:
    ser_sensor = serial.Serial(SERIAL_SENSOR, BAUDRATE, timeout=0.01)
    print(f"Sensor Arduino connected: {SERIAL_SENSOR}")
except Exception as e:
    print(f"Sensor Connection Failed: {e}")
    ser_sensor = None

# ===== 制御パラメータ =====
UP_CH = [1, 2]
DOWN_CH = [5]
MEASURE_DURATION_MS = 140
COOLDOWN_MS = 550
STRONG_RATIO = 5.0
EMA_ALPHA = 0.2
K_SIGMA = 5.9
CALIB_DURATION = 5.0

WEAK_RANGE   = [60.0, 70.0]
STRONG_RANGE = [80.0, 100.0]

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
                    try:
                        line = ser_sensor.readline().decode("utf-8").strip()
                        if line:
                            parts = line.split(",")
                            if len(parts) >= 3:
                                with sensor_lock:
                                    latest_sensor_data = [float(p) for p in parts[:3]]
                    except UnicodeDecodeError:
                        pass
                else:
                    time.sleep(0.001)
            except Exception:
                time.sleep(0.1)
        else:
            time.sleep(0.1)

# ===== 関数群 =====
def compute_rms(buf):
    if len(buf) == 0: return 0
    return np.sqrt(np.mean(np.array(buf) ** 2))

def send_cmd(cmd):
    if ser_motor and ser_motor.is_open:
        try:
            ser_motor.write(cmd.encode("utf-8"))
        except Exception as e:
            print(f"  [CMD] Error: {e}")

def calibrate(duration=CALIB_DURATION):
    global mean, std, calibration_done
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
    print("【設定】強弱の閾値を決定します。")
    
    print(">> 3秒後に『弱い』動作計測...")
    time.sleep(3.0)
    print(">> Start WEAK calibration! (3s)")
    weak_max_val = 0.0
    start_time = time.time()
    while time.time() - start_time < 3.0:
        time.sleep(0.01)
        current_down_level = np.max([ema_val[ch] for ch in DOWN_CH])
        if current_down_level > weak_max_val:
            weak_max_val = current_down_level
    print(f">> Weak Max: {weak_max_val:.2f}")

    print("\n>> 3秒後に『強い』動作計測...")
    time.sleep(3.0)
    print(">> Start STRONG calibration! (3s)")
    strong_max_val = 0.0
    start_time = time.time()
    while time.time() - start_time < 3.0:
        time.sleep(0.01)
        current_down_level = np.max([ema_val[ch] for ch in DOWN_CH])
        if current_down_level > strong_max_val:
            strong_max_val = current_down_level
    print(f">> Strong Max: {strong_max_val:.2f}")

    calibration_done = True
    print("\n=== キャリブレーション完了 ===")

# ===== EMGコールバック =====
def on_emg(emg, movement):
    global current_state, measure_start_time, cooldown_start_time, measure_buffer
    global is_holding_up, ema_val, prev_ema_val

    now = time.time()
    if emg is None: return

    # データ更新
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
    
    down_z_list = [z_scores[ch] for ch in DOWN_CH]
    max_down_z = np.max(down_z_list) if down_z_list else 0

    down_level = np.max([ema_val[ch] for ch in DOWN_CH]) if DOWN_CH else 0

    if current_state == STATE_IDLE:
        # Attack判定
        is_attack_triggered = False
        for ch in DOWN_CH:
            if z_scores[ch] > K_SIGMA:
                is_attack_triggered = True
                break

        if is_attack_triggered:
            send_cmd("I") # 初期動作開始
            current_state = STATE_ATTACKING
            measure_start_time = now
            measure_buffer = []
            is_holding_up = False
            print(f">> ATTACK Triggered (Z: {max_down_z:.2f})")
            return

        # UP判定
        is_up_active = False
        for ch in UP_CH:
            if z_scores[ch] > K_SIGMA:
                is_up_active = True
                break
        
        if is_up_active and not is_holding_up:
            send_cmd("L")
            is_holding_up = True
            print(">> UP! - Maintaining")
        elif not is_up_active and is_holding_up:
            # ★ 削除: send_cmd("R") を削除しました
            # UPを解除しても、明示的な命令は送らず、その位置(UP)をキープします
            is_holding_up = False
            print(">> UP Released (No Return)")

    elif current_state == STATE_ATTACKING:
        measure_buffer.append(down_level)
        if (now - measure_start_time) * 1000 >= MEASURE_DURATION_MS:
            if len(measure_buffer) > 0:
                peak_val = np.max(measure_buffer)
                
                if peak_val >= STRONG_RANGE[0]:
                    print(f"[HIT] STRONG (Raw:{peak_val:.1f})")
                    send_cmd("S")
                else:
                    print(f"[HIT] WEAK (Raw:{peak_val:.1f})")
                    send_cmd("W")

            current_state = STATE_COOLDOWN
            cooldown_start_time = now

    elif current_state == STATE_COOLDOWN:
        if (now - cooldown_start_time) * 1000 >= COOLDOWN_MS:
            # ★ 削除: send_cmd("R") を削除しました
            # クールダウンが終わっても、自動でRestには戻りません（Down位置をキープ）
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
    def myo_worker():
        while is_running:
            m.run()
    
    t_myo = threading.Thread(target=myo_worker, daemon=True)
    t_myo.start()

    # 2. センサースレッド
    if ser_sensor:
        t_sensor = threading.Thread(target=sensor_read_loop, daemon=True)
        t_sensor.start()

    # ★ 削除: send_cmd("R") (初期リセット) を削除
    # 代わりに安全のため初期位置(DOWN)へのコマンドを送るか、Arduino側のsetupに任せます
    # ここでは念のためDOWNを送っておくのも手ですが、ArduinoのSetupでDOWNにしているので省略可
    
    print(">> Calling calibrate()...")
    
    # 3. CSV開始
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

    print("\n==== 制御開始 (Restなし) ====")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        is_running = False
        # ハングを防ぐため、デバイス切断処理(m.disconnectなど)はスキップして
        # CSV保存完了後に強制終了する
        
        if csv_file:
            try:
                csv_file.close()
                print("CSV saved.")
            except Exception as e:
                print(f"CSV close error: {e}")

        print("Force Exit.")
        os._exit(0)

if __name__ == "__main__":
    main()