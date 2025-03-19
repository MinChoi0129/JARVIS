### process.py

import cv2
import time
import numpy as np
import open3d as o3d

import pykinect_azure as pykinect
from modules.parameters import radii1, radii2, radii3, radii4

from modules.body_tracking import (
    get_kinect_body_positions,
    update_body_positions,
    draw_pointing_arrow,
    get_arrow_segment_from_body,
)
from modules.bounding_box_collision import (
    check_cylinder_hit_all_instances_sorted,
)


def process_skeleton(
    vis,
    body_tracker,
    arrow_storage,
    joint_storage,
    instance_boxes,
    pcd,
    labels,
    original_colors,
    prev_instance_ids,  # 이전 프레임에 충돌된 instance ID 리스트 (없으면 None)
):
    """
    스켈레톤 업데이트, arrow 및 pcd 색상 변경을 처리합니다.
    각 radii 리스트(예: radii1, radii2, radii3, radii4)에 대해 충돌 결과를 계산하여,
    충돌된 인스턴스에 대해 pcd 색상을 빨간색으로 갱신하고,
    collision 결과를 딕셔너리로 반환합니다.
    """
    try:
        body_frame = body_tracker.update()
    except Exception as e:
        print("body_tracker.update() 오류:", e)
        return prev_instance_ids, None, None

    prev_instance_name = None
    body_pos = get_kinect_body_positions(body_frame)

    if body_pos is not None:
        update_body_positions(vis, body_pos, joint_storage)
        draw_pointing_arrow(vis, body_pos, arrow_storage)
        start, end = get_arrow_segment_from_body(body_pos)
        if start is not None and end is not None:
            # parameters.py에서 여러 radii 리스트 임포트 (필요에 맞게 값이 정의되어 있어야 함)
            from modules.parameters import radii1, radii2, radii3, radii4

            radii_sets = {
                "radii1": radii1,
                "radii2": radii2,
                "radii3": radii3,
                "radii4": radii4,
            }
            collision_results = {}
            union_ids = set()
            # 각 radii 리스트에 대해 충돌 결과 계산
            for key, radii in radii_sets.items():
                collisions = check_cylinder_hit_all_instances_sorted(
                    start, end, instance_boxes, radii
                )
                if collisions:
                    collided_ids = []
                    collided_names = []
                    for collision in collisions:
                        inst, used_r, t = collision
                        instance_id = inst["instance_label"]
                        instance_name = inst.get("inst_name", "unknown")
                        collided_ids.append(instance_id)
                        collided_names.append(instance_name)
                        union_ids.add(instance_id)
                    collision_results[key] = {
                        "names": collided_names,
                    }
                else:
                    collision_results[key] = {"names": []}
            # pcd 색상 업데이트 (충돌한 인스턴스는 빨간색)
            if union_ids:
                new_colors = original_colors.copy()
                for instance_id in union_ids:
                    mask = labels[:, 0] == instance_id
                    new_colors[mask] = [1, 0, 0]  # 빨간색
                pcd.colors = o3d.utility.Vector3dVector(new_colors)
                vis.update_geometry(pcd)
                arrow_storage["arrow"].paint_uniform_color([1, 0, 0])
                prev_instance_ids = list(union_ids)
                prev_instance_name = collision_results
            else:
                if prev_instance_ids is not None:
                    pcd.colors = o3d.utility.Vector3dVector(original_colors)
                    vis.update_geometry(pcd)
                    prev_instance_ids = None
                arrow_storage["arrow"].paint_uniform_color([1, 1, 0])
                prev_instance_name = collision_results
            print("충돌된 instance (각 반지름별):", collision_results)
        else:
            arrow_storage["arrow"].paint_uniform_color([1, 1, 0])
    else:
        draw_pointing_arrow(vis, None, arrow_storage)

    return prev_instance_ids, prev_instance_name, body_frame


def process_frame(
    device,
    body_tracker,
    vis,
    pcd,
    instance_boxes,
    labels,
    joint_storage,
    arrow_storage,
    original_colors,
    prev_instance_ids,
):
    """
    매 프레임마다 Kinect 디바이스에서 영상을 업데이트하고,
    스켈레톤 및 pcd 색상 처리를 진행한 후 현재 프레임의 컬러 영상을 반환합니다.
    """
    capture = device.update()
    ret, frame = capture.get_color_image()
    ret_depth, depth_frame = capture.get_transformed_colored_depth_image()

    try:
        prev_instance_ids, prev_instance_name, body_frame = process_skeleton(
            vis,
            body_tracker,
            arrow_storage,
            joint_storage,
            instance_boxes,
            pcd,
            labels,
            original_colors,
            prev_instance_ids,
        )
    except Exception as e:
        print("Skeleton 처리 오류:", e)

    if not ret:
        print("Kinect 프레임을 가져올 수 없습니다.")
        return None, prev_instance_ids

    plus_body = body_frame.draw_bodies(frame, pykinect.K4A_CALIBRATION_TYPE_COLOR)
    overlay = cv2.addWeighted(plus_body[:, :, :3], 0.7, depth_frame, 0.3, 0)

    return frame, overlay, prev_instance_ids, prev_instance_name
