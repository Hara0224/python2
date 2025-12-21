import cv2
import mediapipe as mp
import math
import os
import csv
from glob import glob


# === Mediapipe Hands 初期化 ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils


# === 角度計算 ===
def calculate_symmetric_angle(ax, ay, bx, by, cx, cy):
    # ベクトル AB（肘→手首） → 基準
    abx, aby = bx - ax, by - ay
    # ベクトル BC（手首→中指） → 比較
    bcx, bcy = cx - bx, cy - by

    angle_ab = math.atan2(aby, abx)
    angle_bc = math.atan2(bcy, bcx)

    angle_deg = math.degrees(angle_bc - angle_ab)
    angle_360 = (angle_deg + 360) % 360

    # 対称角度（-180〜180°）
    angle_symmetric = ((angle_360 + 180) % 360) - 180
    return angle_symmetric


# === フォルダ内の画像パスを取得 ===
input_folder = r"C:\Users\hrsyn\Desktop\DATAforPython\20250729\img_data_raw22\1_20250729_130913"  # ←画像フォルダに変更
image_paths = sorted(glob(os.path.join(input_folder, "*.jpg")))
output_image_folder = os.path.join(input_folder, "annotated")  # フォルダ名：annotated
os.makedirs(output_image_folder, exist_ok=True)


# === 結果CSVの初期化 ===
output_csv_path = os.path.join(input_folder, "results.csv")
csv_file = open(output_csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["filename", "angle_deg"])

# === 各画像ごとに処理 ===
for image_path in image_paths:
    image = cv2.imread(image_path)
    image_copy = image.copy()
    h, w, _ = image.shape
    elbow_point = []

    # === クリックで肘指定 ===
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            elbow_point.append((x, y))
            print(f"{os.path.basename(image_path)} の肘: {x}, {y}")
            cv2.destroyAllWindows()

    window_name = f"Click Elbow - {os.path.basename(image_path)}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        disp = image_copy.copy()
        if elbow_point:
            cv2.circle(disp, elbow_point[0], 8, (255, 255, 0), -1)
            cv2.putText(
                disp,
                "Elbow",
                (elbow_point[0][0] + 10, elbow_point[0][1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
        cv2.imshow(window_name, disp)
        if cv2.waitKey(1) & 0xFF == 27:
            print("スキップしました")
            elbow_point = []
            break
        if elbow_point:
            break

    if not elbow_point:
        continue  # スキップ

    # === MediaPipe Hands 処理 ===
    image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        wrist = hand_landmarks.landmark[0]
        middle = hand_landmarks.landmark[9]
        wx, wy = int(wrist.x * w), int(wrist.y * h)
        mx, my = int(middle.x * w), int(middle.y * h)
        ex, ey = elbow_point[0]

        # 角度計算
        angle = calculate_symmetric_angle(ex, ey, wx, wy, mx, my)

        # 描画
        cv2.line(image_copy, (ex, ey), (wx, wy), (255, 0, 0), 2)
        cv2.line(image_copy, (wx, wy), (mx, my), (0, 255, 0), 2)
        cv2.putText(
            image_copy,
            f"{angle:.1f} deg",
            (w - 220, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )
        mp_drawing.draw_landmarks(image_copy, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 角度記録
        name_only = os.path.splitext(os.path.basename(image_path))[0]
        csv_writer.writerow([name_only, round(angle, 2)])
        name_only = os.path.splitext(os.path.basename(image_path))[0]
        csv_writer.writerow([name_only, round(angle, 2)])

        # === 線付き画像を保存 ===
        output_image_path = os.path.join(output_image_folder, f"{name_only}_angle.jpg")
        cv2.imwrite(output_image_path, image_copy)

        # 結果表示（確認用）
        cv2.imshow("Result", image_copy)
        cv2.waitKey(500)  # 0にすれば手動で閉じる
        cv2.destroyAllWindows()
    else:
        print(f"{os.path.basename(image_path)}: 手検出できず")
        name_only = os.path.splitext(os.path.basename(image_path))[0]
        csv_writer.writerow([name_only, "N/A"])


# === 終了処理 ===
csv_file.close()
print(f"\n✅ 完了しました：{output_csv_path}")
