import cv2
import mediapipe as mp
import serial
import time
import playsound
import threading
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
i = 0

# Arduinoとのシリアル通信セットアップ
try:
    arduino = serial.Serial("COM4", 9600, timeout=None)  # COMポートとボーレートを設定
    time.sleep(2)  # 接続の安定化
except Exception as e:
    print(f"Arduino Connection Error: {e}")
    exit()

# Webカメラから入力
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(1)

# 音声再生フラグ
sound_playing = False


# 音声再生用の関数
def play_sound():
    global sound_playing
    sound_playing = True
    playsound.playsound("y2mate.com - Gundam Warning  Sound FX HD.mp3")
    # arduino.write(b'1')
    sound_playing = False


with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # 画像処理
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # 顔が検出されているか
        face_detected = results.multi_face_landmarks is not None

        # 音声再生（顔が検出されない場合のみトリガー）、カウントを１増やす
        if not face_detected and not sound_playing:
            print("顔を伏せました")
            i = i + 1
            threading.Thread(target=play_sound, daemon=True).start()

        if i > 2:  # Arduinoに１を送り、カウントを０に戻す
            print("霧吹きが起動します")
            arduino.write(b"1")
            i = 0

        # カメラ画像に情報を描画
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.putText(
            image,
            f"Face Detected: {face_detected}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if face_detected else (0, 0, 255),
            2,
        )

        # メッシュ描画
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )

        # 画像を表示
        cv2.imshow("MediaPipe Face Mesh - Face Detection", image)

        # 終了条件
        if cv2.waitKey(5) & 0xFF in [27, ord("q")]:
            break

# リソース解放
cap.release()
cv2.destroyAllWindows()
arduino.close()
