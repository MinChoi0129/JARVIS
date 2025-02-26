### point_cloud_loader.py

import numpy as np
import open3d as o3d
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# 63개의 랜덤 컬러맵 선택
color_map_names = random.sample(plt.colormaps(), 63)

precomputed_colors = []
for cmap_name in color_map_names:
    cmap = plt.get_cmap(cmap_name)
    safe_colors = []
    for i in range(63):
        color = cmap(i / 62)[:3]
        # 흰색(1,1,1)이나 검정(0,0,0)에 너무 가깝거나, 빨강과 너무 가까운 색은 배제(예시 기준)
        if not (
            np.allclose(color, [1, 1, 1], atol=0.1)
            or np.allclose(color, [0, 0, 0], atol=0.1)
            or np.allclose(color, [1, 0, 0], atol=0.3)
        ):
            safe_colors.append(color)
    if len(safe_colors) < 63:
        safe_colors = (
            safe_colors * (63 // len(safe_colors))
            + safe_colors[: 63 % len(safe_colors)]
        )
    precomputed_colors.append(safe_colors[:63])

names = {
    "ceiling": 0,
    "floor": 1,
    "wall": 2,
    "beam": 3,
    "column": 4,
    "window": 5,
    "door": 6,
    "table": 7,
    "chair": 8,
    "sofa": 9,
    "bookcase": 10,
    "board": 11,
    "clutter": 12,
}

eng_to_kor = {
    "ceiling": "천장",
    "floor": "바닥",
    "wall": "벽",
    "beam": "가로 기둥",
    "column": "세로 기둥",
    "window": "창문",
    "door": "문",
    "table": "책상",
    "chair": "의자",
    "sofa": "소파",
    "bookcase": "책장",
    "board": "칠판",
    "clutter": "기타",
}


def to_korean(eng_label):
    return eng_to_kor[eng_label]


def load_point_cloud_from_instance_npy(file_path, pred_path):
    points = np.load(file_path)  # (N, 3)
    labels = np.load(pred_path)  # (N, 2) 형태라고 가정 -> [class_id, instance_id]

    # 제외할 클래스ID 목록
    exclude_labels = [
        names["ceiling"],
        names["floor"],
        names["wall"],
        names["beam"],
        names["window"],
        names["door"],
        names["clutter"],
    ]
    # labels[:, 0] = class_id
    mask = ~np.isin(labels[:, 0], exclude_labels)

    points = points[mask]
    labels = labels[mask]

    # 색상 할당
    # labels[i, 1] -> 인스턴스 ID
    colors = np.array([precomputed_colors[label[1] % 63][label[1]] for label in labels])

    # 스케일 및 평행이동
    scale = 600
    points = points * scale - np.mean(points, axis=0) + np.array([0, 400, 350])

    # PointCloud 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 인스턴스별 AABB 생성
    unique_instances = np.unique(labels[:, 1])
    instance_boxes = []
    for inst_label in tqdm(unique_instances):
        mask_inst = labels[:, 1] == inst_label
        inst_points = points[mask_inst]
        if len(inst_points) == 0:
            continue

        # IQR 필터링으로 이상치 제거
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

        inst_colors = colors[mask_inst]
        instance_color = inst_colors.mean(axis=0)

        # 클래스 이름
        class_id = labels[mask_inst][0, 0]  # 해당 인스턴스 중 첫 번째 점의 클래스 ID
        class_name = [key for key, value in names.items() if value == class_id][0]

        instance_boxes.append(
            {
                "instance_label": int(inst_label),
                "class_name": class_name,
                "kor_class_name": to_korean(class_name),
                "min_bound": min_bound,
                "max_bound": max_bound,
                "color": instance_color.tolist(),
            }
        )

    # (수정) labels 배열도 함께 반환 (pcd, instance_boxes, labels)
    return pcd, instance_boxes, labels
