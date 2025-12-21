import cv2
import mediapipe as mp
import math

# MediaPipe Hands 初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(x1, y1, x2, y2):
    """手首→中指基部のベクトルが水平線と成す角度を計算"""
    dx = x2 - x1
    dy = y2 - y1
    radians = math.atan2(-dy, -dx)  # OpenCVはY軸が下向きなので -dy
    angle = math.degrees(radians)
    return angle


# 入力画像のパス（★差し替えてください）
image_path = r"C:\Users\hrsyn\Desktop\DATAforPython\20250729\img_data_raw11\1_20250729_123655\0842.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = hands.process(image_rgb)

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # 手首(0)と中指の付け根(9)を取得
        wrist = hand_landmarks.landmark[0]
        middle_base = hand_landmarks.landmark[9]

        h, w, _ = image.shape
        x1, y1 = int(wrist.x * w), int(wrist.y * h)
        x2, y2 = int(middle_base.x * w), int(middle_base.y * h)

        # 角度計算
        angle = calculate_angle(x1, y1, x2, y2)

        # 結果描画
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(image, f"{angle:.1f} deg", (255, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

else:
    print("手が検出されませんでした")

# 結果表示
cv2.imshow("Hand Angle (Wrist to Finger)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
