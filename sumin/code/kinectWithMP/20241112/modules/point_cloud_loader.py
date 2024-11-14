import numpy as np
import open3d as o3d
import random
import matplotlib.pyplot as plt


# 62개의 랜덤 컬러맵 선택
color_map_names = random.sample(plt.colormaps(), 63)

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

# 각 컬러맵에서 62개의 색상 (r, g, b) 값 미리 생성하여 캐시
precomputed_colors = []
for cmap_name in color_map_names:
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i / 62)[:3] for i in range(63)]  # 0부터 61까지의 색상 생성
    precomputed_colors.append(colors)


def load_point_cloud_from_instance_npy(file_path, pred_path, mode):
    """포인트 클라우드 데이터를 npy 파일로부터 로드하고, 인스턴스별 구체 객체를 포함하여 반환."""
    # 포인트와 라벨 데이터 로드
    points = np.load(file_path)
    labels = np.load(pred_path)
    print("PCD, label의 shape:", points.shape, labels.shape)

    # 벽(2), 천장(0) 등의 클래스를 제거하기 위한 마스크 설정
    exclude_labels = [0, 2, 12, 5, 6, 3]
    mask = ~np.isin(
        labels[:, 0], exclude_labels
    )  # 첫 번째 채널(클래스 라벨) 기준으로 필터링

    # 필터링된 포인트와 라벨 데이터
    points = points[mask]
    labels = labels[mask]

    # 색상 매핑 - 인스턴스 라벨에 대한 색상 할당
    colors = np.array(
        [precomputed_colors[label[1] % 63][label[1]] for label in labels]
    )  # 인스턴스 라벨 기준 색상

    print("Filtered label 종류:", len(set(labels[:, 1])))  # 인스턴스 라벨 종류 출력

    # 스케일 조정
    scale = 250 if mode == "object" else 1000
    points = points * scale - np.mean(points, axis=0) + np.array([0, 400, 350])

    # 포인트 클라우드 객체 생성 및 설정
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 인스턴스별 평균 위치에 검은색 구체(sphere) 생성
    instance_spheres = []
    unique_instances = np.unique(labels[:, 1])

    for instance_label in unique_instances:
        # 현재 인스턴스에 해당하는 포인트들의 마스크 생성
        instance_mask = labels[:, 1] == instance_label
        instance_points = points[instance_mask]

        # 평균 좌표 계산
        mean_point = instance_points.mean(axis=0)

        # 구체 객체 생성 및 위치 지정
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=25)
        sphere.translate(mean_point)  # 평균 좌표로 이동
        sphere.paint_uniform_color([0, 0, 0])  # 검은색 설정

        # 생성된 구체 추가
        instance_spheres.append(sphere)

    return pcd, instance_spheres


def load_point_cloud_from_txt(file_path, mode):
    point_cloud_data = np.loadtxt(file_path)

    scale = 100 if mode == "object" else 1000
    points = point_cloud_data[:, :3] * scale
    colors = point_cloud_data[:, 3:] / 255.0

    center = np.mean(points, axis=0)
    points -= center
    points += np.array([84, 84, 168])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd
