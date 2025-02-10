import numpy as np


class SimilarityTransformer:
    def __init__(self):
        self.initial_left_hand = None  # 초기 왼손 좌표
        self.initial_right_hand = None  # 초기 오른손 좌표
        self.initialized = False  # 초기화 여부

    def reset(self):
        self.initial_left_hand = None
        self.initial_right_hand = None
        self.initialized = False
        print("Transformer reset.")

    def compute_similarity_transform(self, left_hand_pos, right_hand_pos):
        """
        초기 좌표를 기준으로 현재 좌표와의 유사 변환 행렬을 계산합니다.
        """
        if not self.initialized:
            # 초기 상태 저장
            self.initial_left_hand = np.array(left_hand_pos)
            self.initial_right_hand = np.array(right_hand_pos)
            self.initialized = True
            print("Initial hand positions set.")
            return np.eye(4)  # 초기에는 단위 행렬 반환

        # 이전 좌표와 현재 좌표 설정
        prev_points = np.array([self.initial_left_hand, self.initial_right_hand])
        current_points = np.array([left_hand_pos, right_hand_pos])

        # Step 1: Calculate scale
        d_prev = np.linalg.norm(prev_points[1] - prev_points[0])
        d_current = np.linalg.norm(current_points[1] - current_points[0])
        s = d_current / d_prev if d_prev != 0 else 1

        # Step 2: Calculate rotation
        v_prev = prev_points[1] - prev_points[0]
        v_current = current_points[1] - current_points[0]
        v_prev = (
            v_prev / np.linalg.norm(v_prev) if np.linalg.norm(v_prev) != 0 else v_prev
        )
        v_current = (
            v_current / np.linalg.norm(v_current)
            if np.linalg.norm(v_current) != 0
            else v_current
        )

        axis = np.cross(v_prev, v_current)
        axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) != 0 else axis
        angle = np.arccos(np.clip(np.dot(v_prev, v_current), -1.0, 1.0))

        # Rodrigues' rotation formula
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        # Step 3: Calculate translation
        C_prev = np.mean(prev_points, axis=0)
        C_current = np.mean(current_points, axis=0)
        t = C_current - s * R @ C_prev

        # Step 4: Build the 4x4 transformation matrix
        M = np.eye(4)
        M[:3, :3] = s * R
        M[:3, 3] = t

        return M
