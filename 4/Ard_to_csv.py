import serial
import csv

SERIAL_PORT = "COM7"
BAUDRATE = 115200
OUTPUT_CSV = "acc_data.csv"

ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "ACC1_val", "ACC1_trig", "ACC2_val", "ACC2_trig"])

    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if not line or "Timestamp" in line:
                continue
            parts = line.split(",")
            if len(parts) != 5:
                continue
            timestamp = int(parts[0])
            acc1_val = int(parts[1])
            acc1_trig = int(parts[2])
            acc2_val = int(parts[3])
            acc2_trig = int(parts[4])
            writer.writerow([timestamp, acc1_val, acc1_trig, acc2_val, acc2_trig])
            print(timestamp, acc1_val, acc1_trig, acc2_val, acc2_trig)
        except KeyboardInterrupt:
            print("終了")
            break
