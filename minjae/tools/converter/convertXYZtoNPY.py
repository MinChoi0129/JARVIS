import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# 파일 경로 설정
xyz_file = "xyzrgb/xyz_raw/702.xyz"
coord_file = "xyzrgb/xyz_preprocessed/702_coord.npy"
color_file = "xyzrgb/xyz_preprocessed/702_color.npy"
normal_file = "xyzrgb/xyz_preprocessed/702_normal.npy"

# .xyz 파일 로딩
data = np.loadtxt(xyz_file, delimiter=",")
coords = data[:, :3].astype(np.float32)
colors = data[:, 3:].astype(np.int32)


# 3D 노멀 벡터 계산
def compute_normals(points, k=10):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(points)
    normals = np.zeros_like(points)

    for i, point in enumerate(points):
        print(i)
        _, indices = nbrs.kneighbors([point])
        neighbors = points[indices[0]]

        # PCA를 통해 노멀 벡터를 계산
        pca = PCA(n_components=3)
        pca.fit(neighbors)
        normal = pca.components_[:, 2]  # 노멀 벡터는 PCA의 3번째 성분으로 추출
        normals[i] = normal

    return normals


# numpy 파일 저장
np.save(coord_file, coords)
print("DONE")
np.save(color_file, colors)
print("DONE")
normals = compute_normals(coords)
np.save(normal_file, normals)

print(f"Saved coord.npy, color.npy, normal.npy to disk.")
