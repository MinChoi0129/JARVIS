import cv2
import mediapipe as mp
import numpy as np
import torch
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
import sys
import os

# Add Gesture_Recognition folder to Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
)

<<<<<<< Updated upstream

# 모델 로드
# GPU에서 학습한 내용을 CPU로 할당
=======
# 모델 로드
>>>>>>> Stashed changes
model = torch.load(
    r"models\model.pt",
    map_location=torch.device("cpu"),
)
model.eval()

<<<<<<< Updated upstream

gesture = {0: "Palm", 1: "Fist"}

actions = ["Palm", "Fist"]
=======
gesture = {0: "Palm", 1: "Fist", 2: "Finger Tip"}
actions = ["Palm", "Fist", "Finger Tip"]
>>>>>>> Stashed changes
seq_length = 30

# MediaPipe 솔루션 초기화
mp_hands = mp.solutions.hands
<<<<<<< Updated upstream
mp_drawing = mp.solutions.drawing_utils
=======
>>>>>>> Stashed changes
hands = mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# 전역 변수로 양손의 시퀀스 데이터 및 제스처 추적 관리
seqs = {"left": [], "right": []}
action_seqs = {"left": [], "right": []}


def process_hand_gestures(color_image):
    global seqs, action_seqs

<<<<<<< Updated upstream
    # axis 1 상하반전
    color_image = np.flip(color_image, axis=1)
=======
    # 1이 좌우반전
    color_image = cv2.flip(color_image, 1)
>>>>>>> Stashed changes

    # BGR에서 RGB로 변환
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # MediaPipe Hand Detection 및 랜드마크 추출
    result = hands.process(rgb_image)
    if not result.multi_hand_landmarks:
        return {"left": None, "right": None}

    handedness_dict = {"Right": "right", "Left": "left"}
    detected_hands = {"left": None, "right": None}

    # 양손 처리
    for hand_landmarks, hand_label in zip(
        result.multi_hand_landmarks, result.multi_handedness
    ):
        handedness = handedness_dict[hand_label.classification[0].label]

        joint = np.zeros((21, 4))
        for j, lm in enumerate(hand_landmarks.landmark):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

        # 인접 랜드마크 간 벡터 계산
        v1 = joint[
            [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3
        ]
        v2 = joint[
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3
        ]
        v = v2 - v1
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        # 각도 계산 및 피처 배열 생성
        angle = np.arccos(
            np.einsum(
                "nt,nt->n",
                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :],
            )
        )
        angle = np.degrees(angle)
        d = np.concatenate([joint.flatten(), angle])

        seqs[handedness].append(d)

        if len(seqs[handedness]) < seq_length:
            continue

        input_data = np.expand_dims(
            np.array(seqs[handedness][-seq_length:], dtype=np.float32), axis=0
        )
        input_data = torch.FloatTensor(input_data)

        # torch 모델을 사용하여 제스처 예측
        y_pred = model(input_data)
        values, indices = torch.max(y_pred.data, dim=1, keepdim=True)

        conf = values.item()

        # Confidence가 0.8 미만이면 인식하지 않음
        if conf < 0.8:
            continue

        action = actions[indices.item()]
        action_seqs[handedness].append(action)

        if len(action_seqs[handedness]) < 2:
            continue

        # 마지막 2개의 제스처가 동일해야 인식
<<<<<<< Updated upstream
        this_action = "None"
=======
        this_action = "?"
>>>>>>> Stashed changes
        if action_seqs[handedness][-1] == action_seqs[handedness][-2]:
            this_action = action

        # 각 손의 제스처 저장
        detected_hands[handedness] = this_action

<<<<<<< Updated upstream
    return detected_hands["left"], detected_hands["right"]
=======
    return detected_hands
>>>>>>> Stashed changes
