import serial
import time

# Arduino設定 (V19と同じ設定)
SERIAL_PORT = "COM4"
BAUDRATE = 115200

def main():
    print(f"Connecting to {SERIAL_PORT} at {BAUDRATE}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1.0)
        # Arduinoのリセット待ち (DTRが働いてリセットされる場合があるため)
        print("Waiting for Arduino restart (2s)...")
        time.sleep(2.0)
    except Exception as e:
        print(f"Connection Failed: {e}")
        return

    print("=== Motor Test Start ===")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            # UP
            cmd = "L"
            print(f"Sending: {cmd} (Lift)")
            ser.write(cmd.encode('utf-8'))
            ser.flush()
            time.sleep(2.0)

            # DOWN / RESET
            cmd = "R"
            print(f"Sending: {cmd} (Release)")
            ser.write(cmd.encode('utf-8'))
            ser.flush()
            time.sleep(2.0)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ser.close()
        print("Closed.")

if __name__ == "__main__":
    main()
