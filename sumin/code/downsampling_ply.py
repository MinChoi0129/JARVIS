import open3d as o3d

# PLY 파일 로드
pcd = o3d.io.read_point_cloud("../pcd_data/my/702.ply")

# 다운샘플링 (여기서 voxel_size는 원하는 해상도에 맞게 설정)
down_pcd = pcd.voxel_down_sample(voxel_size=0.05)

# 다운샘플된 파일 저장
o3d.io.write_point_cloud("downsampled_file.ply", down_pcd)
