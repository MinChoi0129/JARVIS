import numpy as np
import open3d as o3d

file_paths = [
    "/Users/leeminjae0129/Desktop/jarvis/minjae/PointceptS3DISfor702/npy_pred/instance_coord_npy/chair_0.npy",
    "/Users/leeminjae0129/Desktop/jarvis/minjae/PointceptS3DISfor702/npy_pred/instance_coord_npy/chair_1.npy",
    "/Users/leeminjae0129/Desktop/jarvis/minjae/PointceptS3DISfor702/npy_pred/instance_coord_npy/chair_2.npy",
    "/Users/leeminjae0129/Desktop/jarvis/minjae/PointceptS3DISfor702/npy_pred/instance_coord_npy/chair_3.npy",
    "/Users/leeminjae0129/Desktop/jarvis/minjae/PointceptS3DISfor702/npy_pred/instance_coord_npy/chair_4.npy",
    "/Users/leeminjae0129/Desktop/jarvis/minjae/PointceptS3DISfor702/npy_pred/instance_coord_npy/chair_5.npy",
    "/Users/leeminjae0129/Desktop/jarvis/minjae/PointceptS3DISfor702/npy_pred/instance_coord_npy/chair_6.npy",
    # "/Use
    #
    # \
    # rs/leeminjae0129/Desktop/jarvis/minjae/PointceptS3DISfor702/npy_pred/instance_coord_npy/bookcase_0.npy",
    # "/Users/leeminjae0129/Desktop/jarvis/minjae/PointceptS3DISfor702/npy_pred/instance_coord_npy/bookcase_1.npy",
    # "/Users/leeminjae0129/Desktop/jarvis/minjae/PointceptS3DISfor702/npy_pred/instance_coord_npy/clutter_8.npy",
]


coords = []
colors = []

for path in file_paths:
    instance = np.load(path)
    coord = instance[:, :3].tolist()
    color = (instance[:, 3:] / 255).tolist()
    coords += coord
    colors += color


# Open3D 포인트 클라우드 객체 생성
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coords)  # 좌표 설정
pcd.colors = o3d.utility.Vector3dVector(colors)

# 포인트 클라우드 시각화
o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud Visualization")
