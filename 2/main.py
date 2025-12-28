import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルの読み込み
filename = 'sensor_test.csv'

try:
    df = pd.read_csv(filename)
    
    # 時間の計算: マイクロ秒を秒に変換し、開始時間を0秒に合わせる
    # 最初の行の時間を基準(0)とする
    start_time = df['Arduino_Micros'].iloc[0]
    df['Time_sec'] = (df['Arduino_Micros'] - start_time) / 1_000_000

    # グラフの作成
    plt.figure(figsize=(12, 6)) # 横長で見やすく設定
    
    plt.plot(df['Time_sec'], df['Sensor1'], label='Sensor 1', linewidth=1, alpha=0.8)
    plt.plot(df['Time_sec'], df['Sensor2'], label='Sensor 2', linewidth=1, alpha=0.8)

    plt.title('Vibration Sensor Data')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Sensor Value (0-1023)')
    plt.legend()       # 凡例を表示
    plt.grid(True)     # グリッド線を表示
    
    # グラフを表示
    plt.show()

except FileNotFoundError:
    print(f"ファイル '{filename}' が見つかりません。同じフォルダに置いてください。")
except Exception as e:
    print(f"エラーが発生しました: {e}")