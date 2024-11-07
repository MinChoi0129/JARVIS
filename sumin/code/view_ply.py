import open3d as o3d

# PLY 파일 로드
file_path = "../pcd_data/my/702.ply"
pcd = o3d.io.read_point_cloud(file_path)

# 파일 정보 출력
print(pcd)

# 포인트 클라우드 데이터 시각화
o3d.visualization.draw_geometries([pcd])

# 포인트 클라우드 좌표 및 색상 데이터 출력
points = pcd.points
colors = pcd.colors

# 첫 번째 5개 포인트 출력
print("Points:")
print(points[:5])
print("Colors:")
print(colors[:5])
