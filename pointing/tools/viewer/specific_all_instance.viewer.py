import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# 데이터 불러오기
coord = np.load("data/702_coord.npy")  # (n, 3) 형태의 좌표
pred = np.load("data/702_instance_pred.npy")  # (n, ) 형태의 라벨
instance_pred = np.load(
    "data/702_instance_with_class_pred.npy"
)  # (n, ) 형태의 인스턴스 구분된 라벨

# 사용자에게 선택할 라벨 받기 (예: '의자'라벨이 5번이라 가정)
target_label = int(input("시각화할 라벨을 입력하세요: "))

# 선택한 라벨에 해당하는 인덱스 필터링
target_indices = np.where(pred == target_label)[0]

# 나머지 점들 검은색으로 설정
colors = np.zeros((coord.shape[0], 3))  # 기본 검정색

# cmap 설정 (cmap20)
cmap = plt.get_cmap("tab20")
unique_instance_labels = np.unique(instance_pred[target_indices])

# 선택한 라벨의 인스턴스별로 색상 할당
for i, instance_label in enumerate(unique_instance_labels):
    instance_indices = target_indices[instance_pred[target_indices] == instance_label]
    colors[instance_indices] = cmap(i % 20)[:3]  # RGB 값 할당

# open3d 시각화
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coord)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd], window_name="Instance Visualization")
