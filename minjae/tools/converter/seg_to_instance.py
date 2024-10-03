import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

# 데이터 로드
coords = np.load("xyzrgb/xyz_preprocessed/702_coord.npy")  # nx3 좌표 데이터
labels = np.load("npy_pred/702_pred.npy")  # n, 라벨 데이터
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

# 클러스터링 제외 대상 라벨
exclude_labels = [0, 1, 2]

# 각 라벨에 대한 클러스터 개수 설정
cluster_targets = {
    0: 1,
    1: 1,
    2: 4,
    4: 1,
    7: 4,
    8: 11,
    9: 1,
    10: 4,
    12: 40,
}

# 결과 저장할 배열
instance_labels = np.zeros_like(labels)

# 클러스터 번호를 증가시킬 변수
cluster_offset = 0

# 6(door)과 11(board)를 하나의 인스턴스로 처리
shared_instance_label = cluster_offset
cluster_offset += 1  # door와 board를 공유할 클러스터 번호 미리 할당

# 각 라벨별로 클러스터링 수행
for label, n_clusters in tqdm(cluster_targets.items()):
    # 클러스터링 제외 대상인 경우, 그대로 유지
    if label in exclude_labels:
        instance_labels[labels == label] = label
        continue

    # door(6)와 board(11)는 하나의 인스턴스로 간주
    if label in [6, 11]:
        instance_labels[labels == label] = shared_instance_label
        continue

    # 클러스터링 대상에 해당하는 데이터 선택
    mask = labels == label
    points = coords[mask]

    # KMeans 클러스터링 수행
    print(f"{names[label]}에 대한 클러스터링 into {n_clusters}.")
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42).fit(points)

    # 클러스터링 결과를 저장 (고유 클러스터 번호로 조정)
    instance_labels[mask] = kmeans.labels_ + cluster_offset

    # 클러스터 번호 증가
    cluster_offset += n_clusters

print(instance_labels.shape)
print(len(set(instance_labels.flatten())))

# 결과 저장
np.save("npy_pred/702_instance_pred.npy", instance_labels)
