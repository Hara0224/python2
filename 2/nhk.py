import cv2
import numpy as np
import pyautogui

# カメラ初期化
cap = cv2.VideoCapture(1)  # 0はデフォルトカメラ

# 動画保存の設定
width, height = pyautogui.size()
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (width, height))

while True:
    # 画面キャプチャ
    screen = pyautogui.screenshot()
    screen = np.array(screen)
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

    # カメラ映像取得
    ret, frame = cap.read()
    if ret:
        # カメラ映像を小さくして右下に配置
        frame_small = cv2.resize(frame, (200, 150))
        screen[-150:, -200:] = frame_small

    # 録画
    out.write(screen)

    # 表示
    cv2.imshow("Screen & Camera", screen)
    if cv2.waitKey(1) == ord("q"):
        break

# 終了処理
cap.release()
out.release()
cv2.destroyAllWindows()
