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

# ... (接続処理は変更なし) ...
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
UP_CH = [1,2]
DOWN_CH = [5]
MEASURE_DURATION_MS = 150
COOLDOWN_MS = 500
STRONG_RATIO = 5.0
EMA_ALPHA = 0.2  # 0.2 -> 0.3 に上げて反応速度アップ
K_SIGMA = 5.7  # new3.7.pyと同じ感度
CALIB_DURATION = 5.0
# ★強弱の範囲設定 (Min, Max)
WEAK_RANGE   = [60.0, 70.0]
STRONG_RANGE = [80.0, 100.0]

# ===== グローバル変数 =====
rms_buf = [deque(maxlen=30) for _ in range(8)]
ema_val = np.zeros(8)
prev_ema_val = np.zeros(8)

# ★再追加: CSV関連
csv_file = None
csv_writer = None
is_running = True  # 全スレッド停止用フラグ

# 閾値格納用 (Z-scoreベースに変更)
mean = np.zeros(8)
std = np.ones(8)
strong_threshold = 999.0 # これは維持 (Attack後の強弱判定用)
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
            print(f"  [CMD] Sending: {cmd}")
            ser_motor.write(cmd.encode("utf-8"))
            ser_motor.flush() # 確実送信
        except Exception as e:
            print(f"  [CMD] Error: {e}")


def calibrate(duration=CALIB_DURATION):
    # (ロジックは元のままなので省略、ただしCSV保存はQueue経由になるため影響なし)
    # ★new3.7.py 風のキャリブレーション (Mean/Std取得)
    global mean, std, calibration_done, strong_threshold
    print("\n=== キャリブレーション: 3秒間 脱力してください ===")
    
    # データ収集用バッファ
    cal_cols = [[] for _ in range(8)]
    
    start_time = time.time()
    while time.time() - start_time < duration:
        # バックグラウンドスレッドが ema_val を更新しているのでそれをサンプリング
        time.sleep(0.01) 
        for ch in range(8):
            cal_cols[ch].append(ema_val[ch])

    # 平均・標準偏差の計算
    for ch in range(8):
        arr = np.array(cal_cols[ch])
        mean[ch] = np.mean(arr)
        std[ch] = np.std(arr, ddof=1)
        if std[ch] < 1e-6 or np.isnan(std[ch]):
            std[ch] = 1e-6
            
    # Strong Thresholdの計算 (Interactive / Automatic)
    print("\n----------------------------------------")
    print("【設定】強弱の閾値を決定します。")
    print("3秒後に『弱い』動作の計測を開始します...")
    time.sleep(3.0)
    
    print(">> Start WEAK calibration! (3 seconds)")
    weak_max_val = 0.0
    start_time = time.time()
    while time.time() - start_time < 3.0:
        time.sleep(0.01)
        # 現在のダウンチャンネルの最大値を取得
        current_down_level = np.max([ema_val[ch] for ch in DOWN_CH])
        if current_down_level > weak_max_val:
            weak_max_val = current_down_level
    print(f">> 弱い動作の最大値: {weak_max_val:.2f}")

    print("\n続いて3秒後に『強い』動作の計測を開始します...")
    time.sleep(3.0)
    
    print(">> Start STRONG calibration! (3 seconds)")
    strong_max_val = 0.0
    start_time = time.time()
    while time.time() - start_time < 3.0:
        time.sleep(0.01)
        current_down_level = np.max([ema_val[ch] for ch in DOWN_CH])
        if current_down_level > strong_max_val:
            strong_max_val = current_down_level
    print(f">> 強い動作の最大値: {strong_max_val:.2f}")

    calibration_done = True
    print("\n=== 完了 ===")
    print(f"★設定範囲: Weak={WEAK_RANGE}, Strong={STRONG_RANGE}")
    print(f"DEBUG: Mean: {mean}")
    print(f"DEBUG: Std:  {std}")
    print(f"DEBUG: Strong Thr: {strong_threshold:.2f} (Weak:{weak_max_val:.1f} / Strong:{strong_max_val:.1f})")


# ===== メインコールバック =====
def on_emg(emg, movement):
    global current_state, measure_start_time, cooldown_start_time, measure_buffer
    global is_holding_up, ema_val, prev_ema_val

    # ★処理開始時間を記録 (これをCSVのタイムスタンプにする)
    now = time.time()

    if emg is None:
        return

    # 動作確認用ハートビート (1000回に1回表示)


    # ★動作確認用: カウンターで確実に表示 (200Hz想定)
    # nowを使わず、呼び出し回数で判定する
    log_counter = getattr(on_emg, "counter", 0) + 1
    on_emg.counter = log_counter
    
    if log_counter % 100 == 0: # 0.5秒に1回 (200Hz / 2 = 100)
       ch5_val = ema_val[5]
       ch6_val = ema_val[6]
       ch5_slope = ch5_val - prev_ema_val[5]
       ch6_slope = ch6_val - prev_ema_val[6]
       # 値と変化量を表示して、波形が来ているか確認
       print(f"[Status] CH5:{ch5_val:.1f} CH6:{ch6_val:.1f}")

    prev_ema_val = np.copy(ema_val)
    for ch in range(8):
        rms_buf[ch].append(emg[ch])
        rms = compute_rms(rms_buf[ch])
        ema_val[ch] = EMA_ALPHA * rms + (1 - EMA_ALPHA) * ema_val[ch]

    # ★再追加: CSV書き込み
    if csv_writer:
        try:
             # Arduinoのデータなどがない場合は0埋め等対応が必要だが、変数があればそれを使う
             # current_sensor_data は sensor_read_loop で更新されている
             log_data = [now] + ema_val.tolist() + latest_sensor_data + [current_state]
             csv_writer.writerow(log_data)
        except Exception as e:
            pass


    if not calibration_done:
        return

    # --- 以下、制御ロジック (元のまま) ---
    # Z-score計算
    z_scores = (ema_val - mean) / (std + 1e-6)

    # 制御ロジック (Z-score ベース)
    # Attack (DOWN) の判定
    down_z_list = [z_scores[ch] for ch in DOWN_CH]
    max_down_z = np.max(down_z_list)
    
    # UP (Lift) の判定
    up_z_list = [z_scores[ch] for ch in UP_CH]
    max_up_z = np.max(up_z_list)

    # 従来のLevel計算（State判定用）
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
            print(f">> ATTACK! (Z-Score: {max_down_z:.2f})")
            return

        # Debug
        if max_down_z > 2.0: 
             ch_idx = np.argmax(down_z_list)
             target_ch = DOWN_CH[ch_idx]
             #print(f"[DEBUG] Z:{max_down_z:.2f} / Thr:{K_SIGMA} (ch{target_ch+1})")

        # UP Trigger
        is_up_active = False
        for ch in UP_CH:
            if z_scores[ch] > K_SIGMA:
                is_up_active = True
                break
        
        if is_up_active and not is_holding_up:
            send_cmd("L")
            is_holding_up = True
            print(f">> UP! (Z-Score: {max_up_z:.2f}) - Maintaining")
        elif not is_up_active and is_holding_up:
            send_cmd("R")
            is_holding_up = False
            print(">> UP Released.")

    elif current_state == STATE_ATTACKING:
        measure_buffer.append(down_level)
        if (now - measure_start_time) * 1000 >= MEASURE_DURATION_MS:
            if len(measure_buffer) > 0:
                peak_val = np.max(measure_buffer)
                
                # ★判定ロジック (Clamping実装)
                # 優先度: Strong判定 (Min以上ならStrongとみなす)
                if peak_val >= STRONG_RANGE[0]:
                    # Strong判定
                    # 範囲を超えていたら(上)、Minに丸める (下は条件で弾いているのであり得ないはずだが念のため)
                    measured_val = peak_val
                    if measured_val > STRONG_RANGE[1]:
                        measured_val = STRONG_RANGE[0]
                        print(f"[HIT] STRONG (Clamped) (Raw:{peak_val:.1f} -> {measured_val:.1f})")
                    else:
                        print(f"[HIT] STRONG (Raw:{peak_val:.1f})")
                    send_cmd("S")

                else:
                    # Weak判定 (Strong未満はすべてWeak扱い)
                    # 範囲外(下 または 上[Gap])なら Minに丸める
                    measured_val = peak_val
                    if measured_val < WEAK_RANGE[0] or measured_val > WEAK_RANGE[1]:
                        measured_val = WEAK_RANGE[0]
                        print(f"[HIT] weak   (Clamped) (Raw:{peak_val:.1f} -> {measured_val:.1f})")
                    else:
                        print(f"[HIT] weak   (Raw:{peak_val:.1f})")
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
    print(">> Starting Myo thread...")
    def myo_worker():
        while is_running:
            m.run()
    
    t_myo = threading.Thread(target=myo_worker, daemon=True)
    t_myo.start()
    print(">> Myo thread started (Loop Mode).")

    # 2. センサースレッド
    if ser_sensor:
        t_sensor = threading.Thread(target=sensor_read_loop, daemon=True)
        t_sensor.start()


    print(">> Reset motor...")
    send_cmd("R")
    time.sleep(1.0)
    print(">> Motor Test: LIFT (L) - Should Move! <<<")
    send_cmd("L")
    time.sleep(1.0)
    print(">> Motor Test: Release (R)")
    send_cmd("R")


    
    print(">> Calling calibrate()...")
    
    # 3. ★ロギング開始 (キャリブレーション直前にファイル作成)
    save_dir = r"C:\Users\hrsyn\Desktop\masterPY\1"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = os.path.join(save_dir, datetime.datetime.now().strftime("emg_data_%Y%m%d_%H%M%S.csv"))
    
    global csv_file, csv_writer
    try:
        csv_file = open(filename, "w", newline="")
        csv_writer = csv.writer(csv_file)
        # ヘッダー: timestamp, ch1..8, ard_micros, vib1_z, vib2_z, state
        header = ["timestamp"] + [f"ch{i+1}" for i in range(8)] + ["ard_micros", "vib1_z", "vib2_z", "state"]
        csv_writer.writerow(header)
        print(f"CSV Recording started: {filename}")
    except Exception as e:
        print(f"Failed to open CSV: {e}")

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
        
        # CSVファイルを閉じる
        if csv_file:
            csv_file.close()
            print("CSV Closed.")
        print("Exiting immediately...")
        os._exit(0)



if __name__ == "__main__":
    main()
