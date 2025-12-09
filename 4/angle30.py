import cv2
import numpy as np
import math
import os
from glob import glob


def calculate_angle_from_line(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)


# 対象フォルダのパス（必要に応じて変更）
folder_path = r"C:\Users\AZUKI\Desktop\python\EMG2\img_data_high\2up_20250721_113005"
output_folder = os.path.join(folder_path, "results")
os.makedirs(output_folder, exist_ok=True)

# 対象画像をすべて取得（.jpgファイル）
image_paths = sorted(glob(os.path.join(folder_path, "*.jpg")))

for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is None:
        print(f"画像が読み込めません: {image_path}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        longest_line = max(lines, key=lambda line_val: np.hypot(line_val[0][2] - line_val[0][0], line_val[0][3] - line_val[0][1]))
        x1, y1, x2, y2 = longest_line[0]

        angle = calculate_angle_from_line(x1, y1, x2, y2)
        print(f"{os.path.basename(image_path)}: {angle:.2f} 度")

        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(image, (x1, y1), 5, (0, 0, 255), -1)
        cv2.circle(image, (x2, y2), 5, (255, 0, 255), -1)

        text = f"{int(angle)} deg"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        x_text = image.shape[1] - text_size[0] - 10
        y_text = 30
        cv2.putText(image, text, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
        cv2.putText(
            image,
            text,
            (x_text, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            1,
        )
    else:
        print(f"{os.path.basename(image_path)}: 直線が検出されませんでした。")

    # 結果画像を保存
    result_filename = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(result_filename, image)

print("すべての画像を処理しました。結果は 'results' フォルダに保存されました。")
