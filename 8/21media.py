import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# 角度を計算する関数
def calculate_angle(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot = np.dot(unit_v1, unit_v2)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


# カメラ起動
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark

            # ランドマークの画像座標
            p0 = np.array([lm[0].x * w, lm[0].y * h])  # 手首
            p5 = np.array([lm[5].x * w, lm[5].y * h])  # 人差し指基部
            p6 = np.array([lm[6].x * w, lm[6].y * h])  # 人差し指中節
            p17 = np.array([lm[17].x * w, lm[17].y * h])  # 小指の付け根

            # 中間点とその垂直方向にずらした点（親指側）
            v = p6 - p5
            mid56 = (p5 + p6) / 2
            v_perp = np.array([-v[1], v[0]])  # 垂直ベクトル（時計回り90度）
            v_perp = v_perp / np.linalg.norm(v_perp)
            offset = -60  # ピクセル単位で親指方向にずらす
            offset_point = mid56 + v_perp * offset

            # ベクトル定義と角度
            base_vec = p17  # 手首 → 小指の付け根（基準ベクトル）
            stick_vec = offset_point  # オフセット点 → 小指の付け根（スティック方向）

            angle = calculate_angle(base_vec, stick_vec)

            # 描画
            cv2.putText(frame, f"Angle: {angle:.2f} deg", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.circle(frame, tuple(p0.astype(int)), 6, (255, 255, 0), -1)  # 手首
            cv2.circle(frame, tuple(p17.astype(int)), 6, (0, 255, 0), -1)  # 小指の付け根
            cv2.circle(frame, tuple(offset_point.astype(int)), 6, (0, 0, 255), -1)  # オフセット点
            cv2.line(frame, tuple(offset_point.astype(int)), tuple(p17.astype(int)), (0, 255, 255), 2)
            cv2.line(frame, tuple(p0.astype(int)), tuple(p17.astype(int)), (255, 0, 255), 2)

            # ランドマーク全体描画
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Stick Angle Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
