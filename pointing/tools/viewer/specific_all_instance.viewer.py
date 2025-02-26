import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# 데이터 불러오기
coord = np.load("data/702_coord.npy")  # (n, 3) 형태의 좌표
pred = np.load("data/702_instance_pred.npy")  # (n, ) 형태의 라벨
instance_pred = np.load(
    "data/702_instance_with_class_pred.npy"
)  # (n, 2) 형태: [클래스, 인스턴스 ID]

# 사용자에게 선택할 라벨 입력 (예: '의자'라벨이 5번이라고 가정)
target_label = int(input("시각화할 라벨을 입력하세요: "))

# 선택한 클래스의 인덱스 필터링 (pred 배열 이용)
target_indices = np.where(pred == target_label)[0]

# 전체 포인트를 기본 검정색으로 초기화
colors = np.zeros((coord.shape[0], 3))

# instance_pred의 두번째 열을 인스턴스 구분 정보로 사용
target_instance_ids = instance_pred[target_indices, 1]
unique_instance_labels = np.unique(target_instance_ids)

# colormap 설정 (tab20 사용)
cmap = plt.get_cmap("tab20")

# 선택한 클래스 내 각 인스턴스별로 색상 할당
for i, instance_label in enumerate(unique_instance_labels):
    # target_indices 중에서 두번째 열(instance ID)이 현재 인스턴스와 일치하는 포인트 필터링
    instance_mask = target_instance_ids == instance_label
    instance_indices = target_indices[instance_mask]
    colors[instance_indices] = cmap(i % 20)[:3]  # RGB 값 할당

# open3d 시각화
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coord)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd], window_name="Instance Visualization")
