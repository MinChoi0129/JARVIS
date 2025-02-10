import numpy as np
import open3d as o3d


def expand_aabb(
    aabb: o3d.geometry.AxisAlignedBoundingBox, r: float
) -> o3d.geometry.AxisAlignedBoundingBox:
    """
    AABB의 min/max를 각각 r만큼 확장한 새로운 AABB를 반환
    """
    new_aabb = o3d.geometry.AxisAlignedBoundingBox()

    min_pt = aabb.get_min_bound() - np.array([r, r, r])
    max_pt = aabb.get_max_bound() + np.array([r, r, r])

    new_aabb.min_bound = min_pt
    new_aabb.max_bound = max_pt

    return new_aabb


def intersect_segment_aabb(
    start: np.ndarray, end: np.ndarray, aabb: o3d.geometry.AxisAlignedBoundingBox
) -> bool:
    """
    선분( start->end )이 axis aligned bounding box 와 교차하면 True 반환.

    - 슬래브(slab) 방법으로 Ray/Segment vs AABB 충돌을 검사한다.
    - 선분 파라미터 t ∈ [0, 1] 범위 안에서 교차가 생기면 True.
    """
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()

    d = end - start
    tmin, tmax = 0.0, 1.0  # 선분에 대한 t 범위

    for i in range(3):
        if abs(d[i]) < 1e-12:
            # 선분이 축 i에 평행
            if start[i] < min_bound[i] or start[i] > max_bound[i]:
                return False  # 범위 밖이면 교차 불가
        else:
            t1 = (min_bound[i] - start[i]) / d[i]
            t2 = (max_bound[i] - start[i]) / d[i]
            tmin_i, tmax_i = min(t1, t2), max(t1, t2)

            if tmax < tmin_i or tmin > tmax_i:
                return False
            tmin = max(tmin, tmin_i)
            tmax = min(tmax, tmax_i)
            if tmax < tmin:
                return False

    # 여기까지 왔다면 0 <= tmin <= tmax <= 1 사이가 존재할 가능성이 있음
    return tmax >= 0.0 and tmin <= 1.0


def check_cylinder_hit_instances(
    start: np.ndarray,
    end: np.ndarray,
    cylinder_radius: float,
    instance_boxes: list,
):
    """
    - start, end: 포인팅 실린더 축(선분)
    - cylinder_radius: 실린더 반경
    - instance_boxes: [{"instance_label": int, "aabb": AABB}, ...] 구조의 리스트

    선분 vs AABB 충돌 검사 (AABB를 r만큼 확장) 후,
    충돌하는 instance_label 목록을 반환.
    """
    from .bounding_box_collision import expand_aabb, intersect_segment_aabb

    hit_labels = []

    for inst in instance_boxes:
        label = inst["instance_label"]
        aabb = inst["aabb"]

        # AABB 확장(두께 고려)
        expanded_aabb = expand_aabb(aabb, cylinder_radius)

        # 교차 검사
        if intersect_segment_aabb(start, end, expanded_aabb):
            hit_labels.append(label)

    return hit_labels


def create_aabb_lineset(aabb: o3d.geometry.AxisAlignedBoundingBox, color=[1, 0, 0]):
    """
    AABB를 LineSet으로 만들어 반환하는 함수.
    - color: [r, g, b] 형태(0~1)
    """
    # 1) 8개 꼭짓점 가져오기
    corners = aabb.get_box_points()  # o3d.utility.Vector3dVector
    corners_np = np.asarray(corners)  # Numpy 변환 (8,3)

    # 2) 12개 모서리 정의
    #    각 꼭짓점 인덱스는 get_box_points()에서 [0..7] 순으로
    #    대략 아래처럼 대응됩니다.
    #    0,1,2,3 = 아래사각형 / 4,5,6,7 = 위사각형
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],  # 아래 사각형
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],  # 위 사각형
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # 수직 모서리
    ]

    # 3) LineSet 생성
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_np)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(
        [(i[0] * 30, i[1] * 30, i[0] * 30) for i in lines]
    )

    return line_set
