import numpy as np
import open3d as o3d
import random

# 데이터 로드
coord = np.load("data/702_coord.npy")  # (n, 3) 좌표 데이터
pred = np.load("data/702_instance_pred.npy")  # (n,) 라벨 데이터

# clutter (라벨 12) 포인트만 선택
clutter_indices = np.where(pred == 6)[0]
clutter_coord = coord[clutter_indices]

# clutter 포인트 클라우드 생성
clutter_pcd = o3d.geometry.PointCloud()
clutter_pcd.points = o3d.utility.Vector3dVector(clutter_coord)

# DBSCAN 클러스터링을 사용해 clutter 내에서 서로 다른 instance를 구분
# eps와 min_points 값은 데이터에 따라 적절히 조절하세요.
cluster_labels = np.array(
    clutter_pcd.cluster_dbscan(eps=0.05, min_points=10, print_progress=True)
)

# 노이즈(-1)는 제외하고, 클러스터 개수 계산
n_clusters = cluster_labels.max() + 1
print(f"Found {n_clusters} clutter instances (excluding noise)")

# 각 클러스터에 대해 랜덤 색상 부여 (노이즈는 검정색으로 설정)
colors = np.zeros((len(cluster_labels), 3))
for i in range(n_clusters):
    colors[cluster_labels == i] = np.random.rand(3)
colors[cluster_labels == -1] = [0, 0, 0]  # 노이즈에 대한 색상 (검정색)

clutter_pcd.colors = o3d.utility.Vector3dVector(colors)

# 시각화: 각각의 clutter instance가 다른 색상으로 표시됨
o3d.visualization.draw_geometries(
    [clutter_pcd], window_name="Clutter Instances Visualization"
)
