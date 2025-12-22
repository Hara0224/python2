import time
import numpy as np
from collections import deque
from pyomyo import Myo, emg_mode
import serial
import threading

# ===== Arduino設定 =====
#SERIAL_MOTOR = "COM5"  # 環境に合わせて変更してください
#BAUDRATE = 115200

try:
    ser_motor = serial.Serial(SERIAL_MOTOR, BAUDRATE, timeout=0.01)
except Exception as e:
    print(f"Error opening serial: {e}")
    ser_motor = None

# ===== EMGパラメータ =====
FS = 200.0

# ユーザー指定: RMS Windowsize = 50
# Myoは200Hzなので、50サンプル = 0.25秒(250ms)の移動平均となります
RMS_WIN =25

# EMA係数 (RMSが既に平滑化されているため、EMAは補助的に使用)
EMA_ALPHA = 0.5 

# ===== 計測・判定パラメータ =====
MEASURE_DURATION_MS = 50  # トリガー後にデータを計測する期間
COOLDOWN_MS = 150         # 判定後の不感帯

# --- 閾値設定 (RMS値ベース) ---
# 1. 動作開始トリガー (RMS値がこれを超えたら計測開始)
#    キャリブレーションで決定するため、初期値は仮置き
# TRIGGER_RMS = 25  # (削除: 個別閾値 trigger_thresholds を使用)

# キャリブレーション用定数
K_SIGMA = 5.4  # ノイズフロアの標準偏差の何倍を閾値とするか


# 2. 強弱判定の閾値 (RMS値)
#    グラフでの分析結果「50」を設定
STRONG_THRESHOLD_RMS = 50.0 

# 出力値 (Arduinoへ送る値 0-254)
OUT_WEAK = 130   # 弱打時のサーボ強度
OUT_STRONG = 250 # 強打時のサーボ強度

# ===== 状態管理用定数 =====
STATE_IDLE = 0
STATE_MEASURING = 1
STATE_COOLDOWN = 2

# ===== グローバル変数 =====
rms_buf = [deque(maxlen=RMS_WIN) for _ in range(8)]
# RMS値を格納する配列
current_rms_values = np.zeros(8)

# 状態管理変数
current_state = STATE_IDLE
measure_start_time = 0
cooldown_start_time = 0
measure_buffer = []  # 計測値を貯めるリスト

# チャンネルごとのトリガー閾値 (初期値は安全のため高めに設定)
trigger_thresholds = np.zeros(8) + 999.0 


# チャンネル設定
up_ch = [1, 2]      # 上げる動作
down_ch = [5, 6]    # 叩く動作

calibration_done = False

# ===== EMG処理関数 =====
def compute_rms(buf):
    if len(buf) == 0: return 0
    arr = np.array(buf)
    return np.sqrt(np.mean(arr**2))

def calibrate(duration=3.0):
    global calibration_done
    print("\n=== キャリブレーション開始: 3秒間リラックスしてください ===")
    time.sleep(duration) 
    
    global trigger_thresholds
    
    # RMSのバックグラウンドノイズレベルを確認（参考用）
    print("... データ収集中 ...")
    samples = []
    for _ in range(50):
        samples.append(np.copy(current_rms_values))
        time.sleep(0.01)
    
    samples = np.array(samples)
    
    # チャンネルごとの平均と標準偏差を計算
    noise_mean = np.mean(samples, axis=0)
    noise_std = np.std(samples, axis=0)
    
    # 閾値を決定: Mean + K_SIGMA * Std
    trigger_thresholds = noise_mean + K_SIGMA * noise_std
    
    calibration_done = True
    print(f"=== 完了 ===")
    print(f"Noise Mean: {np.round(noise_mean, 2)}")
    print(f"Noise Std : {np.round(noise_std, 2)}")
    print(f"Calculated Thresholds (Mean + {K_SIGMA}sigma):")
    print(np.round(trigger_thresholds, 2))
    print(f"Strong Threshold (RMS):  > {STRONG_THRESHOLD_RMS}")



# ===== EMGハンドラ (メインロジック) =====
def on_emg(emg, movement):
    global calibration_done
    global current_rms_values
    global current_state, measure_start_time, cooldown_start_time, measure_buffer
    
    if emg is None: return

    # 1. RMS計算 (Windowsize=50)
    for ch in range(8):
        rms_buf[ch].append(emg[ch])
        # ここで Windowsize=50 のRMSが計算される
        rms = compute_rms(rms_buf[ch])
        current_rms_values[ch] = rms

    if not calibration_done: return

    # 2. 値の取得
    #    ダウン系チャンネルの中で最大のRMS値を使用
    down_intensity_rms = np.max([current_rms_values[ch] for ch in down_ch])
    up_intensity_rms = np.max([current_rms_values[ch] for ch in up_ch])

    now = time.time()

    # 3. ステートマシンによる制御
    if current_state == STATE_IDLE:
        # トリガー判定 (チャンネルごとの閾値を使用)
        # down_chのいずれかが閾値を超え、かつ up_chのすべてが閾値を下回っている場合
        
        is_down_triggered = False
        for ch in down_ch:
            if current_rms_values[ch] > trigger_thresholds[ch]:
                is_down_triggered = True
                break
        
        is_up_quiet = True
        for ch in up_ch:
            if current_rms_values[ch] > trigger_thresholds[ch]:
                is_up_quiet = False
                break

        # 上げる筋肉が反応していない(誤検知防止) かつ 叩く筋肉がトリガーを超えたら開始
        if is_down_triggered and is_up_quiet:
            current_state = STATE_MEASURING
            measure_start_time = now
            measure_buffer = [] 

    elif current_state == STATE_MEASURING:
        # 計測中: RMS値を貯める
        measure_buffer.append(down_intensity_rms)
        
        # 指定時間が経過したら判定
        if (now - measure_start_time) * 1000 >= MEASURE_DURATION_MS:
            if len(measure_buffer) > 0:
                # 計測期間中の平均RMS値を計算
                avg_rms_val = np.mean(measure_buffer)
                
                final_output = 0
                
                # 平均RMS値が 50 を超えているかどうかで判定
                if avg_rms_val > STRONG_THRESHOLD_RMS:
                    final_output = OUT_STRONG
                    print(f"[ACTION] STRONG (RMS: {avg_rms_val:.1f})")
                else:
                    final_output = OUT_WEAK
                    print(f"[ACTION] WEAK   (RMS: {avg_rms_val:.1f})")

                
                if ser_motor and ser_motor.is_open:
                    ser_motor.write(bytes([final_output]))

            current_state = STATE_COOLDOWN
            cooldown_start_time = now

    elif current_state == STATE_COOLDOWN:
        if (now - cooldown_start_time) * 1000 >= COOLDOWN_MS:
            # 値が落ち着くまで戻らない (すべてのdown_chが閾値を下回るまで)
            all_down_quiet = True
            for ch in down_ch:
                if current_rms_values[ch] > trigger_thresholds[ch]:
                    all_down_quiet = False
                    break
            
            if all_down_quiet: 
                current_state = STATE_IDLE

# ===== メイン処理 =====
def main():
    print("Myo接続中...")
    m = Myo(mode=emg_mode.RAW)
    m.connect()
    m.add_emg_handler(on_emg)
    m.set_leds([0, 0, 255], [0, 0, 255])
    m.vibrate(1)
    time.sleep(1)
    
    # データ受信用スレッド
    def worker():
        try:
            while True: m.run()
        except: pass
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    
    try:
        calibrate()
        
        print(f"\n=== ドラム制御: RMS閾値判定モード (Window={RMS_WIN}, Thr={STRONG_THRESHOLD_RMS}) ===")
        print("Ctrl+C で終了")
        
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting...")
    finally:
        try:
            m.disconnect()
        except: pass
        
        if ser_motor and ser_motor.is_open:
            try:
                ser_motor.close()
            except: pass
        print("Disconnected.")

if __name__ == "__main__":
    main()