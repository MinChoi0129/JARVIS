import numpy as np
import open3d as o3d

# 데이터 로드
coord = np.load("data/702_coord.npy")  # (n, 3) 좌표 데이터
pred = np.load("data/702_instance_with_class_pred.npy")  # (n,) 라벨 데이터

# 관심 있는 클래스
target_label = 12

# 색상 설정 (beam은 빨간색, 나머지는 검은색)
colors = np.array(
    [[1, 1, 0] if label == target_label else [0, 0, 0] for label in pred[:, 0]]
)

# Open3D 포인트 클라우드 객체 생성
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coord)  # 좌표 설정
pcd.colors = o3d.utility.Vector3dVector(colors)  # 색상 설정

# 포인트 클라우드 시각화
o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud Visualization")
