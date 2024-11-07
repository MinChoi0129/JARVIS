import cv2
import mediapipe as mp
import numpy as np
import torch

import sys
import os

# Add Gesture_Recognition folder to Python path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'Gesture_Recognition')))


# 모델 로드
# GPU에서 학습한 내용을 CPU로 할당
model = torch.load(r'C:\\Users\\gamja5th\\Documents\\24_2_pioneer\\code\\Gesture_Recognition\\main_models\\model.pt',
                   map_location=torch.device('cpu'))
model.eval()


gesture = {
    0: 'Palm',
    1: 'Fist',
    2: 'Finger Tip'
}

actions = ['Palm', 'Fist', 'Finger Tip']
seq_length = 30

# MediaPipe 솔루션 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 전역 변수로 제스처 데이터 시퀀스 관리
seq = []
action_seq = []


def process_hand_landmarks(color_image):
    global seq, action_seq

    # BGR에서 RGB로 변환
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    bgr_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)

    # MediaPipe Hand Detection 및 랜드마크 추출
    result = hands.process(rgb_image)

    if not result.multi_hand_landmarks:
        return bgr_image, None

    if result.multi_hand_landmarks:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10,
                        11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                        12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
            v = v2 - v1

            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10,
                                            12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))

            angle = np.degrees(angle)

            d = np.concatenate([joint.flatten(), angle])

            seq.append(d)

            # 손 랜드마크 그리기
            mp_drawing.draw_landmarks(
                bgr_image, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                return bgr_image, result

            input_data = np.expand_dims(
                np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            input_data = torch.FloatTensor(input_data)

            # torch 모델을 사용하여 제스처 예측
            y_pred = model(input_data)
            values, indices = torch.max(y_pred.data, dim=1, keepdim=True)

            conf = values.item()

            # Confidence가 0.9 미만이면 인식하지 않음
            if conf < 0.9:
                continue

            action = actions[indices.item()]
            action_seq.append(action)

            if len(action_seq) < 2:
                continue

            # 마지막 3개의 제스처가 동일하지 않으면 인식되지 않음
            this_action = '?'
            if action_seq[-1] == action_seq[-2]:
                this_action = action
                # print(f"Recognized Gesture: {this_action}")

            # 화면에 제스처 출력
            cv2.putText(bgr_image, f'{this_action.upper()}',
                        org=(int(res.landmark[0].x * bgr_image.shape[1]),
                             int(res.landmark[0].y * bgr_image.shape[0] + 20)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 255, 255), thickness=2)

    return bgr_image, result
