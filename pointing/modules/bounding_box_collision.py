###bounding_box_collision.py

import numpy as np
import open3d as o3d


def expand_aabb(bbox, r):
    new_min = bbox["min_bound"] - np.array([r, r, r])
    new_max = bbox["max_bound"] + np.array([r, r, r])
    return {"min_bound": new_min, "max_bound": new_max}


def intersect_segment_aabb_with_t(start, end, bbox):
    """
    슬래브 기법을 이용하여 선분(start->end)과 AABB 충돌을 검사하고,
    충돌이 발생하면 t (선분의 파라미터, [0,1] 범위) 중 가장 처음 충돌하는 지점의 t값을 반환.
    충돌이 없으면 None 반환.
    """
    min_bound = bbox["min_bound"]
    max_bound = bbox["max_bound"]
    d = end - start
    tmin, tmax = 0.0, 1.0
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if start[i] < min_bound[i] or start[i] > max_bound[i]:
                return None
        else:
            t1 = (min_bound[i] - start[i]) / d[i]
            t2 = (max_bound[i] - start[i]) / d[i]
            tmin_i = min(t1, t2)
            tmax_i = max(t1, t2)
            tmin = max(tmin, tmin_i)
            tmax = min(tmax, tmax_i)
            if tmax < tmin:
                return None
    if tmax >= 0.0 and tmin <= 1.0:
        return tmin
    return None


def check_cylinder_hit_all_instances(start, end, instance_boxes):
    """
    각 인스턴스의 AABB(충돌 두께 고려 확장)를 선분과 충돌 검사하여,
    충돌하는 모든 인스턴스를 리스트로 반환합니다.
    충돌하는 인스턴스가 없으면 빈 리스트를 반환합니다.
    """
    cylinder_radius = 0.05
    hit_instances = []
    for inst in instance_boxes:
        bbox = {"min_bound": inst["min_bound"], "max_bound": inst["max_bound"]}
        expanded_bbox = expand_aabb(bbox, cylinder_radius)
        t = intersect_segment_aabb_with_t(start, end, expanded_bbox)
        if t is not None:
            hit_instances.append(inst)
    return hit_instances


def check_cylinder_hit_all_instances_sorted(start, end, instance_boxes, radii):
    """
    주어진 반지름 리스트(radii)에 대해 각 인스턴스의 AABB를 선분과 충돌 검사합니다.

    충돌이 발생하면 튜플 (inst, r, t)를 hit_instances에 추가합니다.
      - inst: 인스턴스 정보
      - r: 사용된 반지름
      - t: 선분 상 충돌 위치 (0~1, 작을수록 ray 시작에 가까움)

    모든 반지름에 대해 충돌 결과를 기록한 후, 반지름(r) 우선, 그 다음 선분 상의 위치(t) 기준으로 정렬하여 반환합니다.
    """
    hit_instances = []
    for inst in instance_boxes:
        for r in radii:
            bbox = {"min_bound": inst["min_bound"], "max_bound": inst["max_bound"]}
            expanded_bbox = expand_aabb(bbox, r)
            t = intersect_segment_aabb_with_t(start, end, expanded_bbox)
            if t is not None:
                hit_instances.append((inst, r, t))
                break
    hit_instances.sort(key=lambda x: (x[1], x[2]))
    return hit_instances


def create_aabb_lineset(bbox, default_color=[1, 0, 0]):
    min_bound = bbox["min_bound"]
    max_bound = bbox["max_bound"]
    corners = np.array(
        [
            [min_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]],
        ]
    )
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_color = bbox.get("color", default_color)
    line_set.colors = o3d.utility.Vector3dVector(
        [line_color for _ in range(len(lines))]
    )
    return line_set
