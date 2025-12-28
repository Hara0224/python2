import time
import numpy as np
from collections import deque
from pyomyo import Myo, emg_mode
import serial
import threading
import sys
import io

# ===== ログ抑制設定 =====
# pyomyoが出す大量の "data with unknown attr" を消すためのフィルタ
class SuppressFilter:
    def write(self, text):
        if "data with unknown attr" not in text:
            sys.__stdout__.write(text)
    def flush(self):
        sys.__stdout__.flush()

# 標準出力をフィルタ経由にする
sys.stdout = SuppressFilter()

# ===== Arduino設定 =====
SERIAL_MOTOR = "COM6"  # ★Arduinoのポートに合わせて変更してください
BAUDRATE = 115200

try:
    ser_motor = serial.Serial(SERIAL_MOTOR, BAUDRATE, timeout=0.01)
except:
    ser_motor = None

# ===== パラメータ =====
UP_CH = [1, 2]     # 振り上げ (1,2ch)
DOWN_CH = [5, 6]   # 攻撃 (5,6ch)

COOLDOWN_MS = 150  # 動作後の待機時間

# 感度設定
# 1. 動き出しの感度 (変化量)
RISE_THRESHOLD_BASE = 1.0

# 2. 強打判定の感度 (変化量がさらにこの倍率を超えたら強打)
SLOPE_STRONG_RATIO = 2.0 

# 3. 振り上げ維持の感度 (絶対値)
UP_HOLD_SENSITIVITY = 5.0 

EMA_ALPHA = 0.3 

# ===== 変数 =====
rms_buf = [deque(maxlen=30) for _ in range(8)]
ema_val = np.zeros(8)
prev_ema_val = np.zeros(8)

# 閾値
trigger_slope_thr = np.zeros(8) + 999.0 
strong_slope_thr = 999.0                
level_thresholds = np.zeros(8) + 999.0  

calibration_done = False

# 状態
STATE_IDLE = 0
STATE_COOLDOWN = 2

current_state = STATE_IDLE
cooldown_start_time = 0
is_holding_up = False

# ===== 関数 =====
def compute_rms(buf):
    if len(buf) == 0: return 0
    return np.sqrt(np.mean(np.array(buf)**2))

def send_cmd(cmd):
    if ser_motor and ser_motor.is_open:
        try: ser_motor.write(cmd.encode('utf-8'))
        except: pass

def calibrate(duration=3.0):
    global trigger_slope_thr, strong_slope_thr, level_thresholds, calibration_done
    
    # フィルタを一時解除してメッセージを表示
    sys.stdout = sys.__stdout__
    print(f"\n=== キャリブレーション: {duration}秒間 脱力してください ===")
    sys.stdout = SuppressFilter()
    
    time.sleep(duration)
    
    val_samples = []
    diff_samples = []
    
    # データ安定待ち
    for _ in range(10): time.sleep(0.01)
        
    for _ in range(50):
        current_val = np.copy(ema_val)
        current_diff = np.abs(ema_val - prev_ema_val)
        val_samples.append(current_val)
        diff_samples.append(current_diff)
        time.sleep(0.01)
    
    # 統計計算
    val_mean = np.mean(val_samples, axis=0)
    val_std = np.std(val_samples, axis=0)
    level_thresholds = val_mean + (val_std * UP_HOLD_SENSITIVITY)
    level_thresholds = np.maximum(level_thresholds, 15.0)

    diff_mean = np.mean(diff_samples, axis=0)
    diff_std = np.std(diff_samples, axis=0)
    
    trigger_slope_thr = diff_mean + (diff_std * RISE_THRESHOLD_BASE)
    trigger_slope_thr = np.maximum(trigger_slope_thr, 2.0)
    
    base_slope = np.mean(trigger_slope_thr[DOWN_CH])
    strong_slope_thr = base_slope * SLOPE_STRONG_RATIO

    calibration_done = True
    
    sys.stdout = sys.__stdout__
    print("=== 完了 ===")
    print(f"Trigger Slope: {base_slope:.2f}")
    print(f"Strong Slope : {strong_slope_thr:.2f}")
    sys.stdout = SuppressFilter()

def on_emg(emg, movement):
    global current_state, cooldown_start_time, is_holding_up, ema_val, prev_ema_val
    
    if emg is None: return
    
    prev_ema_val = np.copy(ema_val)
    
    for ch in range(8):
        rms_buf[ch].append(emg[ch])
        rms = compute_rms(rms_buf[ch])
        ema_val[ch] = EMA_ALPHA * rms + (1 - EMA_ALPHA) * ema_val[ch]
        
    if not calibration_done: return

    # 特徴量
    up_level = np.max([ema_val[ch] for ch in UP_CH])
    
    # DOWNの傾き
    down_slope_list = [(ema_val[ch] - prev_ema_val[ch]) for ch in DOWN_CH]
    max_slope = np.max(down_slope_list)
    
    now = time.time()

    # --- ロジック ---
    if current_state == STATE_IDLE:
        
        # [A] 攻撃判定: 傾きトリガー
        is_triggered = False
        for i, ch in enumerate(DOWN_CH):
            if down_slope_list[i] > trigger_slope_thr[ch]:
                is_triggered = True
                break
        
        if is_triggered:
            # 瞬時に強弱判定
            if max_slope > strong_slope_thr:
                send_cmd('S')
                # sys.stdout = sys.__stdout__
                # print(f"[HIT] STRONG!! (Slope: {max_slope:.1f})")
                # sys.stdout = SuppressFilter()
            else:
                send_cmd('W')
                # sys.stdout = sys.__stdout__
                # print(f"[HIT] weak...  (Slope: {max_slope:.1f})")
                # sys.stdout = SuppressFilter()
            
            current_state = STATE_COOLDOWN
            cooldown_start_time = now
            is_holding_up = False
            return

        # [B] 振り上げ判定 (維持)
        is_up_active = up_level > np.mean(level_thresholds[UP_CH])
        
        if is_up_active and not is_holding_up:
            send_cmd('L')
            is_holding_up = True
            
        elif not is_up_active and is_holding_up:
            send_cmd('R')
            is_holding_up = False

    elif current_state == STATE_COOLDOWN:
        if (now - cooldown_start_time) * 1000 >= COOLDOWN_MS:
            send_cmd('R')
            current_state = STATE_IDLE
            is_holding_up = False

# ===== メイン処理 =====
def main():
    print("Myo接続中... (USBドングルを挿し直すと安定する場合があります)")
    
    try:
        m = Myo(mode=emg_mode.RAW)
        m.connect()
        
        # ★重要: スレッド開始前に設定を済ませる (競合回避)
        print("初期化中...")
        try:
            m.set_leds([0, 255, 255], [0, 255, 255])
            m.vibrate(1)
        except:
            print("[WARN] LED設定に失敗しましたが続行します")

        m.add_emg_handler(on_emg)
        
        # 初期位置へ
        send_cmd('R')
        
        # ★ここで初めてスレッド開始
        def worker():
            try:
                while True: m.run()
            except: pass
        
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        
        calibrate()
        
        sys.stdout = sys.__stdout__
        print("\n==== 最速応答モード (Slope=Force) ====")
        print(" 立ち上がりの速さで強弱が決まります。")
        print(" Ctrl+C で終了")
        sys.stdout = SuppressFilter()
        
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        sys.stdout = sys.__stdout__
        print(f"エラー発生: {e}")
    finally:
        print("\n終了処理中...")
        try: m.disconnect()
        except: pass
        if ser_motor: ser_motor.close()
        print("Bye.")

if __name__ == "__main__":
    main()