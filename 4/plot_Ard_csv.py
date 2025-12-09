import pandas as pd
import matplotlib.pyplot as plt

# CSV読み込み
df = pd.read_csv("acc_data.csv")
df["Time_s"] = df["Timestamp"] / 1000.0  # ms → 秒

# 任意の値（例: 閾値）
THRESHOLD1 = 620
THRESHOLD2 = 620

plt.figure(figsize=(12, 6))

# ACC1/ACC2の値
plt.plot(df["Time_s"], df["ACC1_val"], label="ACC1_val", color="blue")
plt.plot(df["Time_s"], df["ACC2_val"], label="ACC2_val", color="red")

# トリガー判定
plt.scatter(
    df["Time_s"][df["ACC1_trig"] == 1],
    df["ACC1_val"][df["ACC1_trig"] == 1],
    color="blue",
    marker="o",
    label="ACC1 Trigger",
)
plt.scatter(
    df["Time_s"][df["ACC2_trig"] == 1],
    df["ACC2_val"][df["ACC2_trig"] == 1],
    color="red",
    marker="o",
    label="ACC2 Trigger",
)

# 横線を引く
plt.axhline(y=THRESHOLD1, color="blue", linestyle="--", label="ACC1 Threshold")
plt.axhline(y=THRESHOLD2, color="red", linestyle="--", label="ACC2 Threshold")

plt.xlabel("Time (s)")
plt.ylabel("Acceleration / Trigger")
plt.title("ACC1 & ACC2 Data with Trigger Points and Thresholds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
