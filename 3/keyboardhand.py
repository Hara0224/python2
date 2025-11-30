import serial
import time
import keyboard

# シリアルポートの設定（ポート名は使用している環境に合わせて変更）
ser = serial.Serial("COM14", 9600)
last_pressed = None


def main():
    print("動作開始: 0から7までの数字を押すと動作します。'exit' で終了")

    while True:
        for key in "01234567":  # 0から7までのキーを順にチェック
            if (
                keyboard.is_pressed(key) and last_pressed != key
            ):  # キーが押されていて、前回押されたキーと異なる場合
                print("{} を送信します".format(key))
                ser.write(key.encode())  # 文字をバイトとして送信
                last_pressed = key  # 現在のキーを記録
                time.sleep(0.1)  # 長押しを防ぐための待機

        # escキーが押された場合は終了
        if keyboard.is_pressed("esc"):
            print("終了します")
            ser.write(b"exit")  # Arduinoに終了を送信
            break

        # 0-7のキーがすべて離された場合
        elif not (
            keyboard.is_pressed("0")
            or keyboard.is_pressed("1")
            or keyboard.is_pressed("2")
            or keyboard.is_pressed("3")
            or keyboard.is_pressed("4")
            or keyboard.is_pressed("5")
            or keyboard.is_pressed("6")
            or keyboard.is_pressed("7")
        ):
            last_pressed = None  # すべてのキーが離された時にリセット


if __name__ == "__main__":
    main()
