import open3d as o3d
import numpy as np


def load_point_cloud(file_path):
    # 포인트 클라우드를 로드합니다.
    data = np.loadtxt(file_path)
    return data


def visualize_point_cloud(data):
    # 데이터를 분리합니다.
    points = data[:, 0:3]
    colors = data[:, 3:6] / 255.0

    center = np.mean(points, axis=0)
    print(center)

    # 포인트 클라우드를 생성합니다.
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    center_cloud = o3d.geometry.PointCloud()
    center_cloud.points = o3d.utility.Vector3dVector([center])
    center_cloud.colors = o3d.utility.Vector3dVector([[0, 0, 0]])  # 검정색

    # 중심점을 다른 점들보다 크게 보이게 하기 위해 Sphere로 변환
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=0.03)  # 원하는 크기로 설정
    center_sphere.translate(center)  # 중심점 위치로 이동
    center_sphere.paint_uniform_color([0, 0, 0])  # 검정색

    # 포인트 클라우드와 중심점을 동시에 시각화합니다.
    o3d.visualization.draw_geometries([point_cloud, center_sphere])


# 파일 경로를 지정합니다.
file_path = '../pcd_data/Area_1/conferenceRoom_2/Annotations/chair_20.txt'

# 포인트 클라우드를 로드하고 시각화합니다.
data = load_point_cloud(file_path)
visualize_point_cloud(data)
