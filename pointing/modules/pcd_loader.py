######pcd_loader.py###

import numpy as np
import open3d as o3d
import json
from tqdm import tqdm


names = {
    "nothing": 0,
    "laptop1": 1,
    "laptop2": 2,
    "mouse1": 3,
    "mouse2": 4,
    "phone1": 5,
    "phone2": 6,
    "headphones1": 7,
    "headphones2": 8,
    "tumbler1": 9,
    "tumbler2": 10,
    "tissue": 11,
    "wet wipes": 12,
    "diary1": 13,
    "diary2": 14,
    "pen": 15,
    "cushion": 16,
    "blanket": 17,
    "book": 18,
    "glasses": 19,
    "chair1": 20,
    "chair2": 21,
    "chair3": 22,
    "chair4": 23,
    "table": 24,
}


eng_to_kor = {
    "nothing": "사용하지 않음",
    "laptop1": "노트북1",
    "laptop2": "노트북2",
    "mouse1": "마우스1",
    "mouse2": "마우스2",
    "phone1": "휴대폰1",
    "phone2": "휴대폰2",
    "headphones1": "헤드폰1",
    "headphones2": "헤드폰2",
    "tumbler1": "텀블러1",
    "tumbler2": "텀블러2",
    "tissue": "휴지",
    "wet wipes": "물티슈",
    "diary1": "다이어리1",
    "diary2": "다이어리2",
    "pen": "펜",
    "cushion": "쿠션",
    "blanket": "담요",
    "book": "책",
    "glasses": "안경",
    "chair1": "의자1",
    "chair2": "의자2",
    "chair3": "의자3",
    "chair4": "의자4",
    "table": "책상",
}


def to_korean(eng_label):
    return eng_to_kor[eng_label] if eng_label in eng_to_kor else "알 수 없음"


def load_point_cloud_from_instance_pcd(file_path, json_path):
    """
    file_path: .pcd 파일 경로
    json_path: .json 파일 경로 (data: [[idx,R,G,B] or [R,G,B], ...],
                               colors: [[R,G,B,inst_id] or [R,G,B], ...])
    """
    # ---------------------------
    # 1) .pcd 파일 로드
    # ---------------------------
    pcd = o3d.io.read_point_cloud(file_path)  # open3d.geometry.PointCloud
    points = np.asarray(pcd.points)  # (N, 3)

    # pcd에 컬러 정보가 없는 경우 기본값 (166, 215, 43) 할당 (0~1 스케일)
    if not pcd.has_colors():
        num_points = points.shape[0]
        default_color = np.array(
            [166 / 255.0, 215 / 255.0, 43 / 255.0], dtype=np.float32
        )
        pcd.colors = o3d.utility.Vector3dVector(np.tile(default_color, (num_points, 1)))

    # ---------------------------
    # 2) .json 파일 로드
    # ---------------------------
    with open(json_path, "r") as f:
        json_data = json.load(f)

    # (예: data: [[idx, R, G, B], ...], colors: [[R, G, B, inst_id], ...] )
    data_array = json_data["data"]  # 길이가 N인 배열
    colors_array = json_data["colors"]  # (R, G, B, inst_id)

    # ---------------------------
    # 2-1) colors -> (R, G, B) -> instance_id 딕셔너리
    # ---------------------------
    color_to_instance = {}
    for item in colors_array:
        if len(item) == 4:  # [R, G, B, inst_id]
            r, g, b, inst_id = item
        else:
            print(f"[colors] Unexpected format: {item}")
            continue

        color_to_instance[(r, g, b)] = inst_id

    # ---------------------------
    # 2-2) data -> idx, color 설정
    # ---------------------------
    # num_points = len(data_array)
    num_points = int(135665)
    default_color = np.array([166 / 255.0, 215 / 255.0, 43 / 255.0], dtype=np.float32)
    point_colors = np.tile(default_color, (num_points, 1))
    labels = np.zeros((num_points, 1), dtype=np.int32)  # [class_id, instance_id]

    for i, item in enumerate(data_array):
        if len(item) == 4:  # [idx, R, G, B]
            idx, r, g, b = item
        else:
            print(f"[data] Unexpected format: {item}")
            continue

        try:
            point_colors[idx] = [r / 255.0, g / 255.0, b / 255.0]
        except:
            pass

        # (R,G,B) -> instance_id 매핑
        inst_id = color_to_instance.get((r, g, b), 0)
        labels[idx, 0] = inst_id

    # ---------------------------
    # 3) pcd 좌표 변환/업데이트
    # ---------------------------
    scale = 600
    points = points * scale
    points = points - np.mean(points, axis=0) + np.array([0, 400, 350])

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    # ---------------------------
    # 4) 인스턴스별 AABB 계산
    # ---------------------------

    unique_instances = np.unique(labels[:, 0])
    instance_boxes = []
    for inst_label in tqdm(unique_instances):
        # ★ inst_label이 0이면 스킵 (바운딩 박스 X)
        if inst_label == 0:
            continue

        mask_inst = labels[:, 0] == inst_label
        inst_points = points[mask_inst]
        if len(inst_points) == 0:
            continue

        # IQR 필터링
        q1 = np.percentile(inst_points, 25, axis=0)
        q3 = np.percentile(inst_points, 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered_points = inst_points[
            np.all((inst_points >= lower_bound) & (inst_points <= upper_bound), axis=1)
        ]
        if len(filtered_points) == 0:
            continue

        min_bound = np.min(filtered_points, axis=0)
        max_bound = np.max(filtered_points, axis=0)

        inst_colors = point_colors[mask_inst]
        instance_color = inst_colors.mean(axis=0)

        # inst_id (여기서는 inst_label)를 이용하여 inst_name을 찾음
        inst_name_candidates = [k for k, v in names.items() if v == inst_label]
        if len(inst_name_candidates) == 1:
            inst_name = inst_name_candidates[0]
            kor_inst_name = to_korean(inst_name)
        else:
            inst_name = "unknown"
            kor_inst_name = "알수없음"

        instance_boxes.append(
            {
                "instance_label": int(inst_label),
                "inst_name": inst_name,
                "kor_inst_name": kor_inst_name,
                "min_bound": min_bound.tolist(),
                "max_bound": max_bound.tolist(),
                "color": instance_color.tolist(),
            }
        )

    return pcd, instance_boxes, labels
