import cv2
import math
import mediapipe as mp
import os
import time


def resize(image):

    DESIRED_HEIGHT = 480
    DESIRED_WIDTH = 480

    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))

    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))

    # cv2.imshow("frame", img)
    return img


# Abbreviazioni

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Crea un oggetto GestureRecognizer
base_options = BaseOptions(
    model_asset_path=f"{os.path.dirname(os.path.abspath(__file__))}/model/gesture_recognizer.task",
    delegate=BaseOptions.Delegate.CPU,
)

options = GestureRecognizerOptions(
    base_options=base_options, running_mode=VisionRunningMode.VIDEO
)

recognizer = GestureRecognizer.create_from_options(options)


# 0 -> Droidcam portatile
# 1 -> cam portatile
# 3 -> webcam usb

cap = cv2.VideoCapture(3)

while True:
    ret, frame = cap.read()

    # frame = resize(frame)
    # Flip the image horizontally for a selfie-view display.

    cv2.imshow("Frame", cv2.flip(frame, 1))

    # cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Carica il frame
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Analizza il frame per riconoscere una gesture

    #! NON FUNZIONA SU MACOS
    # recognition_result = recognizer.recognize_for_video(
    #     image, int(cap.get(cv2.CAP_PROP_POS_MSEC))
    # )

    timestamp = int(time.time() * 1000)
    recognition_result = recognizer.recognize(image, timestamp)

    # Se riconoscimento andato a buon fine

    if recognition_result.gestures:
        segno = recognition_result.gestures[0][0]
        mano = recognition_result.handedness[0][0]
        print("*******", segno.category_name, segno.score, end=" ")
        print("*******", mano.display_name, mano.score)

        # multi_hand_landmarks = recognition_result.hand_landmarks[0]
        # print(multi_hand_landmarks)
        # print(recognition_result)


recognizer.close()
cap.release()
cv2.destroyAllWindows()
