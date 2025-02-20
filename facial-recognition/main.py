import cv2
import math
import mediapipe as mp
import os


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
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Crea un oggetto FaceDetector
base_options = BaseOptions(
    model_asset_path=f"{os.path.dirname(os.path.abspath(__file__))}/model/blaze_face_short_range.tflite",
    delegate=BaseOptions.Delegate.CPU,
)

options = FaceDetectorOptions(
    base_options=base_options, running_mode=VisionRunningMode.VIDEO
)

detector = FaceDetector.create_from_options(options)

# 0 -> Droidcam portatile
# 1 -> cam portatile
# 3 -> webcam usb
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # frame = resize(frame)  # Abilita questa linea se vuoi ridimensionare il frame

    print("🤔 ret:", ret)

    if ret:
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow("Frame", cv2.flip(frame, 1))
        # cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Carica il frame
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Analizza il frame per riconoscere un viso
    detection_result = detector.detect_for_video(
        image, int(cap.get(cv2.CAP_PROP_POS_MSEC))
    )

    # Se riconoscimento andato a buon fine
    # print(len(detection_result.detections))
    print("** Rilevati", len(detection_result.detections), "visi")

detector.close()
cap.release()
cv2.destroyAllWindows()
