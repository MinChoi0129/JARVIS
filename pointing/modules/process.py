### process.py

import cv2
import time
import numpy as np
import open3d as o3d

import pykinect_azure as pykinect


from modules.body_tracking import (
    get_kinect_body_positions,
    update_body_positions,
    draw_pointing_arrow,
    get_arrow_segment_from_body,
)
from modules.bounding_box_collision import (
    check_cylinder_hit_all_instances,
    create_aabb_lineset,
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
    스켈레톤 업데이트, arrow 및 포인트 클라우드 색상 변경을 처리합니다.
    충돌된 모든 인스턴스에 대해 pcd 색상을 빨간색으로 갱신하고, arrow 색상을 변경하며,
    충돌된 instance의 ID 목록을 출력합니다.
    """
    body_frame = body_tracker.update()
    body_pos = get_kinect_body_positions(body_frame)
    if body_pos is not None:
        # 스켈레톤 선 및 관절 구체 업데이트
        update_body_positions(vis, body_pos, joint_storage)
        # 팔꿈치-손목 arrow 업데이트
        draw_pointing_arrow(vis, body_pos, arrow_storage)

        # arrow의 시작, 끝, 반경 계산
        start, end, arrow_radius = get_arrow_segment_from_body(body_pos)
        if start is not None and end is not None:
            # 충돌 인스턴스를 모두 검사하여 리스트로 반환
            collision_list = check_cylinder_hit_all_instances(
                start, end, arrow_radius, instance_boxes
            )
            if collision_list:
                collided_instance_ids = []
                new_colors = original_colors.copy()
                # 모든 충돌 인스턴스에 대해 색상 업데이트
                for inst in collision_list:
                    instance_id = inst["instance_label"]
                    collided_instance_ids.append(instance_id)
                    mask = labels[:, 1] == instance_id
                    new_colors[mask] = [1, 0, 0]  # 빨간색
                pcd.colors = o3d.utility.Vector3dVector(new_colors)
                vis.update_geometry(pcd)
                print("충돌된 instance:", collided_instance_ids)
                arrow_storage["arrow"].paint_uniform_color([1, 0, 0])
                prev_instance_ids = collided_instance_ids
            else:
                # 충돌이 없으면 이전 프레임에 충돌된 instance가 있었을 경우 원본 색상 복원
                if prev_instance_ids is not None:
                    pcd.colors = o3d.utility.Vector3dVector(original_colors)
                    vis.update_geometry(pcd)
                    prev_instance_ids = None
                arrow_storage["arrow"].paint_uniform_color([1, 1, 0])
    else:
        # 스켈레톤 정보가 없으면 arrow 숨김 처리
        draw_pointing_arrow(vis, None, arrow_storage)

    return prev_instance_ids, body_frame


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
    prev_instance_id,
):
    """
    매 프레임마다 Kinect 디바이스에서 영상을 업데이트하고,
    스켈레톤 및 pcd 색상 처리를 진행한 후 현재 프레임의 컬러 영상을 반환합니다.
    """
    capture = device.update()
    ret, frame = capture.get_color_image()
    ret_depth, depth_frame = capture.get_transformed_colored_depth_image()

    try:
        prev_instance_id, body_frame = process_skeleton(
            vis,
            body_tracker,
            arrow_storage,
            joint_storage,
            instance_boxes,
            pcd,
            labels,
            original_colors,
            prev_instance_id,
        )
    except Exception as e:
        print("Skeleton 처리 오류:", e)

    if not ret:
        print("Kinect 프레임을 가져올 수 없습니다.")
        return None, prev_instance_id

    plus_body = body_frame.draw_bodies(frame, pykinect.K4A_CALIBRATION_TYPE_COLOR)
    overlay = cv2.addWeighted(plus_body[:, :, :3], 0.7, depth_frame, 0.3, 0)

    return frame, overlay, prev_instance_id
