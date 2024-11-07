import open3d as o3d
import numpy as np


def load_point_cloud(file_path):
    # 포인트 클라우드를 로드합니다.
    data = np.loadtxt(file_path)
    return data


def visualize_point_cloud_with_box(data):
    # 데이터를 분리합니다.
    points = data[:, 0:3]
    colors = data[:, 3:6] / 255.0

    center = np.mean(points, axis=0)

    # 기존 포인트 클라우드 생성
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # 네모 박스 생성 (단위 크기 1x1x1, 좌하단 꼭지점이 (0, 0, 0)에서 시작)
    box = o3d.geometry.TriangleMesh.create_box(
        width=1.0, height=0.1, depth=1.0)  # 크기는 필요에 맞게 수정
    box.translate([-0.5, -0.05, -0.5])  # (0, 0, 0)를 중심으로 박스를 이동
    box.paint_uniform_color([0.9, 0.9, 0.9])  # 박스 색을 회색으로 설정

    # 포인트 클라우드를 박스 위에 띄우기 위해 y 좌표를 조정
    points -= center  # 박스 위로 포인트 클라우드를 조금 띄우기 위해 y축으로 0.1만큼 이동
    points[:, 1] += 0.5
    point_cloud.points = o3d.utility.Vector3dVector(points)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.7, origin=[0, 0, 0])

    # 박스와 포인트 클라우드를 동시에 시각화
    o3d.visualization.draw_geometries([box, point_cloud, coordinate_frame])


# 파일 경로를 지정합니다.
file_path = '../pcd_data/Area_1/conferenceRoom_2/Annotations/chair_20.txt'

# 포인트 클라우드를 로드하고 시각화합니다.
data = load_point_cloud(file_path)
visualize_point_cloud_with_box(data)
