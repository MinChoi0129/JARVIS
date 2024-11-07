import open3d as o3d
import pandas as pd
import numpy as np
import time
import pptk

# 파일 경로
file_path = "../pcd_data/Area_1/office_1/office_1.txt"

# Pandas 사용
df = pd.read_csv(file_path, sep=",", header=None,
                 names=["x", "y", "z", "r", "g", "b"])
points = df[["x", "y", "z"]].to_numpy()
colors = df[["r", "g", "b"]].to_numpy() / 255.0  # 색상을 [0, 1] 범위로 정규화


v = pptk.viewer(points)
v.attributes(colors)
v.set(point_size=0.0001)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])
