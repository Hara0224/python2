import socket
import time
import sys  # ここを追加

# RPi.GPIO関連はすべて削除（PCで実行する場合の最もシンプルな形）
# DMM, 電源の初期設定
# ご使用のTEXIOのIPアドレスとポート番号に合わせて変更してください
TEXIOHost = "172.16.5.133"  # <-- ここにTEXIOのIPアドレスを設定してください
TEXIOPort = (
    2268  # TEXIOが使用するポート番号を設定してください。通常は5025の場合が多いです。
)

# ソケットオブジェクトの作成
TEXIO = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def connect_texio():
    """TEXIO電源装置に接続します。"""
    try:
        print(f"TEXIO ({TEXIOHost}:{TEXIOPort}) に接続を試みます...")
        TEXIO.connect((TEXIOHost, TEXIOPort))

        # TEXIOのIDを問い合わせる（オプション）
        TEXIO.send(b"idn?\n")
        response_texio = TEXIO.recv(4096)
        print(f"TEXIO接続成功: {response_texio.decode().strip()}")

        # 必要に応じて、TEXIOの初期設定コマンドを追加
        # 例: TEXIO.send(b'VOLT 0.0\n') # 初期電圧を0に設定
        # 例: TEXIO.send(b'OUTP OFF\n') # 初期状態では出力をOFFにしておく
        return True
    except Exception as e:
        print("TEXIOへの接続に失敗しました: {e}")
        return False


def set_texio_voltage(voltage):
    """
    TEXIOの電圧を設定し、出力をONにします。
    voltage: 設定したい電圧値 (float)
    """
    try:
        # 電圧設定コマンドを送信
        # '{voltage:.1f}' は電圧値を小数点以下1桁でフォーマットします
        command = f"SOUR:VOLT {str(voltage):.1f}\n".encode()
        TEXIO.send(command)
        print("TEXIOに電圧設定コマンドを送信: {command.decode().strip()}")

        # 出力をONにする
        TEXIO.send(b"OUTP ON\n")
        print("TEXIOの出力をONにしました。")
    except Exception as e:
        print("TEXIOの電圧設定に失敗しました")


def turn_off_texio_output():
    """TEXIOの出力をOFFにします。"""
    try:
        TEXIO.send(b"OUTP OFF\n")
        print("TEXIOの出力を停止しました。")
        # 必要であれば、リセットコマンドなどを追加
        # TEXIO.send(b'RST\n')
    except Exception as e:
        print("TEXIO停止エラー: {e}")


def close_texio_connection():
    """TEXIOとのソケット接続を閉じます。"""
    try:
        TEXIO.close()
        print("TEXIOとの接続を閉じました。")
    except Exception as e:
        print("ソケットクローズエラー: {e}")


# --- メイン処理 ---
if __name__ == "main":
    if not connect_texio():
        print("TEXIOに接続できなかったため、プログラムを終了します。")
        sys.exit(1)  # ここを sys.exit(1) に変更

    try:
        print("\n--- 電圧設定テスト ---")

        # 例1: 10.0Vに設定
        target_voltage_1 = 10.0
        print("TEXIOの電圧を {" + str(target_voltage_1) + "}V に設定します。")
        set_texio_voltage(target_voltage_1)
        time.sleep(5)  # 5秒待機

    except KeyboardInterrupt:
        print("\nプログラムがユーザーによって中断されました。")
    finally:
        # 終了処理
        turn_off_texio_output()
        close_texio_connection()
