import serial
import csv
import time

# --- 設定 ---
COM_PORT = 'COM5'  # Arduinoのポート番号に合わせて変更してください
BAUD_RATE = 115200 # ArduinoのSerial.beginと同じ速度
OUTPUT_FILE = 'sensor_test2.csv'

def save_and_print_data():
    try:
        # Arduinoとの通信開始
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        print(f"接続完了: {COM_PORT}")
        print("Ctrl+C を押すと終了して保存します...")
        print("-" * 40)

        # CSVファイルを開いて準備
        with open(OUTPUT_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            # ヘッダー（項目の名前）を書き込み
            header = ["Arduino_Micros", "Sensor1", "Sensor2"]
            writer.writerow(header)
            print(f"ヘッダー書き込み: {header}")

            # データを読み込み続けるループ
            while True:
                if ser.in_waiting > 0:
                    try:
                        # Arduinoから1行データを受け取る
                        line = ser.readline().decode('utf-8').strip()
                        
                        if line:
                            data = line.split(',')
                            # データが3つ揃っているか確認（ノイズ対策）
                            if len(data) == 3:
                                # 1. ファイルに保存
                                writer.writerow(data)
                                
                                # 2. 画面にも表示（これで動作確認できます）
                                print(f"保存中: 時間={data[0]}, センサ1={data[1]}, センサ2={data[2]}")
                                
                    except ValueError:
                        pass # 文字化けなどは無視

    except serial.SerialException:
        print("エラー: ポートが見つかりません。Arduinoが接続されているか、ポート番号が正しいか確認してください。")
    except KeyboardInterrupt:
        print("\n終了します。")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("通信を閉じました。")
            print(f"ファイル '{OUTPUT_FILE}' を確認してください。")

if __name__ == "__main__":
    save_and_print_data()