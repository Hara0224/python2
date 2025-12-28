import pandas as pd
import matplotlib.pyplot as plt

filename = "sensor_test.csv"

try:
    df = pd.read_csv(filename)

    # 時間の計算: 秒単位、開始を0に
    start_time = df["Arduino_Micros"].iloc[0]
    df["Time_sec"] = (df["Arduino_Micros"] - start_time) / 1_000_000

    # --- ここが追加・変更部分 ---
    # 1. 中心値（平均）を引いて、交流成分（AC）だけにする
    # これで静止時が 0 付近になります
    sensor1_ac = df["Sensor1"] - df["Sensor1"].mean()
    sensor2_ac = df["Sensor2"] - df["Sensor2"].mean()

    # 2. 絶対値（Rectification）を取る
    df["Sensor1_Abs"] = sensor1_ac.abs()
    df["Sensor2_Abs"] = sensor2_ac.abs()
    # -------------------------

    # グラフの作成
    plt.figure(figsize=(12, 6))

    # 絶対値化したデータをプロット
    plt.plot(df["Time_sec"], df["Sensor1_Abs"], label="Sensor 1 (Abs)", linewidth=1, alpha=0.8)
    plt.plot(df["Time_sec"], df["Sensor2_Abs"], label="Sensor 2 (Abs)", linewidth=1, alpha=0.8)

    plt.title("Vibration Magnitude (Absolute Value)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (Abs)")
    plt.legend()
    plt.grid(True)

    plt.show()

except FileNotFoundError:
    print(f"ファイル '{filename}' が見つかりません。")
except Exception as e:
    print(f"エラーが発生しました: {e}")
