import numpy as np
import open3d as o3d


def get_kinect_hand_positions(body_frame, c2w):
    """Kinect에서 손 위치 추적."""
    if body_frame:
        try:
            skeleton = body_frame.get_body_skeleton()
        except SystemExit:
            return False, None, None
        except Exception as e:
            raise Exception("Skeleton Fetching Error")
        lhp = skeleton.joints[8].position.xyz
        rhp = skeleton.joints[15].position.xyz
        left_hand_xyz = np.array([lhp.x, lhp.y, lhp.z, 1]).T
        right_hand_xyz = np.array([rhp.x, rhp.y, rhp.z, 1]).T
        return True, (c2w @ left_hand_xyz)[:3], (c2w @ right_hand_xyz)[:3]
    return False, None, None


def update_hand_positions(
    vis,
    left_hand,
    right_hand,
    line_set,
    left_hand_pos,
    right_hand_pos,
    instance_spheres,
    left_hand_ges,
    right_hand_ges,
):
    """손 위치 및 선 업데이트, 인스턴스 구체 색상 변경."""
    # 손 위치 업데이트
    left_hand.translate((left_hand_pos - np.asarray(left_hand.get_center())))
    right_hand.translate((right_hand_pos - np.asarray(right_hand.get_center())))

    print(left_hand_ges, right_hand_ges)

    # 제스처를 visualization을 위한 손 색상 변화 (fist일 때 색이 진해짐)
    left_hand_color = [1, 1, 1] if left_hand_ges == "fist" else [0.4, 0, 0]
    right_hand_color = [1, 1, 1] if right_hand_ges == "fist" else [0, 0, 0.4]

    left_hand.paint_uniform_color(left_hand_color)
    right_hand.paint_uniform_color(right_hand_color)

    # 손과 손을 잇는 거리 선 업데이트
    new_left_pos, new_right_pos = map(
        np.asarray, (left_hand.get_center(), right_hand.get_center())
    )

    line_set.points = o3d.utility.Vector3dVector([new_left_pos, new_right_pos])

    # 각 구체에 대해 손 위치와의 거리 계산 및 색상 업데이트
    for sphere in instance_spheres:
        sphere_center = np.asarray(sphere.get_center())

        # 왼손과 구체 거리 계산
        left_distance = np.linalg.norm(new_left_pos - sphere_center)
        # 오른손과 구체 거리 계산
        right_distance = np.linalg.norm(new_right_pos - sphere_center)

        # 가까운 거리 판정 (200 미만일 경우 빨간색, 그렇지 않으면 검정색)
        if left_distance < 200 and right_distance < 200:
            sphere.paint_uniform_color([1, 0, 0])  # 빨간색
        else:
            sphere.paint_uniform_color([0, 0, 0])  # 검정색

        # 구체 업데이트
        vis.update_geometry(sphere)

    vis.update_geometry(left_hand)
    vis.update_geometry(right_hand)
    vis.update_geometry(line_set)

    distance = np.linalg.norm(new_left_pos - new_right_pos)
    distance_cm = round(int(distance), -1) // 100 * 10
    return vis, distance_cm
