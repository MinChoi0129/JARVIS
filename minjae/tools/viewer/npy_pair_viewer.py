import numpy as np
import open3d as o3d
import random

# 데이터 로드
coord = np.load("xyzrgb/xyz_preprocessed/702_coord.npy")  # (n, 3) 형태의 좌표 데이터
pred = np.load("npy_pred/702_instance_pred.npy")  # (n,) 형태의 label 데이터

print(pred)

# 고유한 클래스 개수 추출
unique_labels = list(set(pred))

names = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "table",
    8: "chair",
    9: "sofa",
    10: "bookcase",
    11: "board",
    12: "clutter",
}


num_classes = len(unique_labels)
print(f"Number of unique labels: {num_classes}")

# 랜덤한 색상 생성
random_colors = np.array(
    [[random.random(), random.random(), random.random()] for _ in range(num_classes)]
)

# 각 라벨에 색상 매핑
label_to_color = {label: random_colors[i] for i, label in enumerate(unique_labels)}
colors = np.array([label_to_color[label] for label in pred])

# Open3D 포인트 클라우드 객체 생성
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coord)  # 좌표 설정
pcd.colors = o3d.utility.Vector3dVector(colors)  # 색상 설정

# 포인트 클라우드 시각화
o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud Visualization")
