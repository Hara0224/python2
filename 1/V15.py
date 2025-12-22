import time
import numpy as np
from collections import deque
from pyomyo import Myo, emg_mode
import serial
import threading

# ===== Arduino設定 =====
# SERIAL_MOTOR = "COM5"  # 環境に合わせて変更してください
# BAUDRATE = 115200

try:
    ser_motor = serial.Serial(SERIAL_MOTOR, BAUDRATE, timeout=0.01)
except Exception as e:
    # print(f"Error opening serial: {e}")
    ser_motor = None

# ===== パラメータ設定 =====
# チャンネル設定 (Myoの電極番号 0-7)
# ユーザー指定: 上げる=2,3ch / 下げる=5,6ch
UP_CH = [2, 3]
DOWN_CH = [5, 6]

# サーボ角度設定 (0-254)
# 初期位置（下・打面）
POS_DOWN = 10   
# 最大振り上げ位置
POS_UP_MAX = 160 

# 閾値 (キャリブレーションで決定しますが、初期値を設定)
THRESHOLD_UP = 30.0   # これを超えると上がり始める
THRESHOLD_DOWN = 30.0 # これを超えると強制的に振り下ろす（叩きつけ）

# 平滑化用 (動きを滑らかにする)
EMA_ALPHA = 0.3 # 0.0~1.0 小さいほど滑らかだが遅延する

# ===== グローバル変数 =====
rms_buf = [deque(maxlen=30) for _ in range(8)] # 反応速度重視でWindowサイズを少し短く(30)
current_rms = np.zeros(8)
smoothed_val = 0 # サーボ出力の平滑化用
calibration_done = False
trigger_thresholds = np.zeros(8) + 999.0

# ===== 関数群 =====
def compute_rms(buf):
    if len(buf) == 0: return 0
    return np.sqrt(np.mean(np.array(buf)**2))

def calibrate():
    global trigger_thresholds, calibration_done
    print("\n=== キャリブレーション: 3秒間リラックスしてください ===")
    time.sleep(3)
    
    samples = []
    print("... ノイズ計測中 ...")
    for _ in range(50):
        samples.append(np.copy(current_rms))
        time.sleep(0.01)
    
    noise_mean = np.mean(samples, axis=0)
    noise_std = np.std(samples, axis=0)
    
    # ノイズ平均 + 5シグマ を閾値とする
    trigger_thresholds = noise_mean + 5.0 * noise_std
    
    # 安全のため、最低値を保証 (あまりに低いと誤作動するため)
    trigger_thresholds = np.maximum(trigger_thresholds, 20.0)
    
    calibration_done = True
    print(f"=== 完了 ===")
    print(f"Thresholds (UP 2,3ch): {trigger_thresholds[UP_CH]}")

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def on_emg(emg, movement):
    global current_rms, smoothed_val
    
    if emg is None: return
    
    # 1. RMS計算
    for ch in range(8):
        rms_buf[ch].append(emg[ch])
        current_rms[ch] = compute_rms(rms_buf[ch])
        
    if not calibration_done: return

    # 2. 意図の検出
    # 上げる力 (2,3chの最大値)
    up_intensity = np.max([current_rms[ch] for ch in UP_CH])
    # 下げる力 (5,6chの最大値)
    down_intensity = np.max([current_rms[ch] for ch in DOWN_CH])
    
    # 閾値チェック
    up_active = up_intensity > trigger_thresholds[UP_CH].mean()
    down_active = down_intensity > trigger_thresholds[DOWN_CH].mean() + 10 # 下げは少し閾値を高く（誤作動防止）

    target_pos = POS_DOWN # デフォルトは下（重力に従う）

    # 3. 動作ロジック
    if down_active:
        # 下げる筋肉が反応したら、即座に「下」へ (アクセント/叩きつけ)
        target_pos = POS_DOWN
        # 素早く反応させるため、スムージングをリセットして即代入
        smoothed_val = POS_DOWN 
        
    elif up_active:
        # 上げる筋肉が反応したら、その強さに応じて角度を決める（アナログ制御）
        # RMS 30〜150 の範囲を、角度 POS_DOWN〜POS_UP_MAX にマッピング
        # これにより「弱く上げれば低く」「強く上げれば高く」なります
        mapped_angle = map_value(up_intensity, 30, 150, POS_DOWN, POS_UP_MAX)
        
        # 範囲制限
        target_pos = np.clip(mapped_angle, POS_DOWN, POS_UP_MAX)
    
    else:
        # どちらも反応していない時 -> 脱力 -> 下に戻る
        target_pos = POS_DOWN

    # 4. 平滑化と送信 (急激なジッターを抑える)
    # 下げる動作(down_active)の時はキレ良くしたいので平滑化を弱くまたは無視するロジックもアリ
    if down_active:
        out_byte = int(target_pos)
    else:
        smoothed_val = (smoothed_val * (1 - EMA_ALPHA)) + (target_pos * EMA_ALPHA)
        out_byte = int(smoothed_val)

    # Arduinoへ送信
    if ser_motor and ser_motor.is_open:
        # データ量を減らすため、前回と値が変わった時だけ送る等の工夫も可
        ser_motor.write(bytes([out_byte]))
        
        # デバッグ表示（値が大きく変わったときのみ表示など）
        # print(f"UP:{up_intensity:.1f} DOWN:{down_intensity:.1f} -> Angle:{out_byte}")

# ===== メイン処理 =====
def main():
    print("Myo接続中...")
    m = Myo(mode=emg_mode.RAW)
    m.connect()
    m.add_emg_handler(on_emg)
    m.set_leds([0, 255, 0], [0, 255, 0])
    m.vibrate(1)
    
    # モーターを初期位置へ
    if ser_motor:
        ser_motor.write(bytes([POS_DOWN]))
    
    def worker():
        try:
            while True: m.run()
        except: pass
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    
    calibrate()
    
    print("\n=== ドラム制御開始 ===")
    print(" [動作イメージ]")
    print(" 1. 手首を上げる (2,3ch) -> 義手が上がる")
    print(" 2. 力を抜く or 手首を下げる (5,6ch) -> 義手が振り下ろされる")
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        m.disconnect()
        if ser_motor: ser_motor.close()

if __name__ == "__main__":
    main()