import mediapipe as mp
import cv2
import numpy as np
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# MediaPipe 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# 모델 경로 설정 (파일 경로 정확히 확인)
model_path = 'kinectWithMP/gesture_recognizer.task'

if not os.path.exists(model_path):
    print(f"Model file not found at {model_path}")
else:
    with open(model_path, 'rb') as file:
        model_data = file.read()

    # Gesture Recognizer 초기화
    b_o = BaseOptions(model_asset_buffer=model_data)
    options = GestureRecognizerOptions(
        base_options=b_o,
        running_mode=VisionRunningMode.IMAGE
    )
    recognizer = GestureRecognizer.create_from_options(options)
    print("Gesture Recognizer loaded successfully.")


def process_hand_landmarks(color_image):
    # BGR에서 RGB로 변환
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    bgr_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # 손 랜드마크 추출
    landmark_results = hands.process(rgb_image)

    # MediaPipe Image로 변환
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Gesture Recognizer를 사용하여 제스처 인식
    recognition_result = recognizer.recognize(mp_image)
    if recognition_result.gestures:
        top_gesture = recognition_result.gestures[0][0]
        print(
            f"Top gesture: {top_gesture.category_name}, Confidence: {top_gesture.score}")
    else:
        print("No gesture recognized.")

    # 랜드마크가 검출되면 그려서 반환
    if landmark_results.multi_hand_landmarks:
        for hand_landmarks in landmark_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                bgr_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        return bgr_image, landmark_results
    return bgr_image, None
