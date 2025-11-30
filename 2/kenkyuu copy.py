import socket
import time
import sys

TEXIOHost = "172.16.5.133"
TEXIOPort = 2268

TEXIO = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def connect_texio():
    """TEXIO電源装置に接続します。"""
    try:
        print(f"TEXIO ({TEXIOHost}:{TEXIOPort}) に接続を試みます...")
        TEXIO.connect((TEXIOHost, TEXIOPort))

        TEXIO.send(b'idn?\n')
        response_texio = TEXIO.recv(4096)
        print(f"TEXIO接続成功: {response_texio.decode().strip()}")

        return True
    except Exception as e:
        print(f"TEXIOへの接続に失敗しました: {e}")
        return False


def set_texio_voltage(voltage):
    try:
        command = f"SOUR:VOLT {voltage:.1f}\n".encode()
        TEXIO.send(command)
        print(f"{voltage:.1f} V に設定しました。")

        TEXIO.send(b'OUTP ON\n')
        print("TEXIOの出力をONにしました。")
    except Exception as e:
        print(f"TEXIOの電圧設定に失敗しました: {e}")

def turn_off_texio_output():
    try:
        TEXIO.send(b'OUTP OFF\n')
        print("TEXIOの出力を停止しました。")
    except Exception as e:
        print(f"TEXIO停止エラー: {e}")

def close_texio_connection():
    try:
        TEXIO.close()
        print("TEXIOとの接続を閉じました。")
    except Exception as e:
        print(f"ソケットクローズエラー: {e}")

if __name__ == "__main__":
    if not connect_texio():
        print("TEXIOに接続できなかったため、プログラムを終了します。")
        sys.exit(1)

    try:
        print("\n--- 電圧設定テスト ---")
        target_voltage_1 = 10.0
        print(f"TEXIOの電圧を {target_voltage_1}V に設定します。")
        set_texio_voltage(target_voltage_1)
        time.sleep(5)

    except KeyboardInterrupt:
        print("\nプログラムがユーザーによって中断されました。")
    finally:
        turn_off_texio_output()
        close_texio_connection()
