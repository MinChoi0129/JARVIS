###body_tracking.py

import numpy as np
import open3d as o3d
from modules.parameters import C2W, skeleton_edges


def get_kinect_body_positions(body_frame):
    if body_frame is None or body_frame.get_num_bodies() == 0:
        return None
    try:
        skeleton = body_frame.get_body_skeleton(0)
    except Exception as e:
        return None

    joints = []
    for joint in skeleton.joints:
        if joint.confidence_level >= 1:
            pos = joint.position.xyz
            joints.append((C2W @ (np.array([pos.x, pos.y, pos.z, 1]).T))[:3])
        else:
            joints.append([False, False, False])
    return np.array(joints, dtype=np.float32)


def update_body_positions(
    vis, body_pos, storage={"line_set": None, "spheres": None, "spheres_pos": None}
):
    """
    스켈레톤을 선(LineSet)으로 그리고, 각 관절마다 작은 구체(스피어)를 추가하여 위치 업데이트.
    """
    # 선(LineSet) 업데이트
    if body_pos is None or (isinstance(body_pos, np.ndarray) and body_pos.size == 0):
        if storage.get("line_set") is not None:
            try:
                vis.remove_geometry(storage["line_set"], reset_bounding_box=False)
            except:
                pass
            storage["line_set"] = None
        return
    valid_indices = []
    valid_coords = []
    for joint_id, pos in enumerate(body_pos):
        if (
            isinstance(pos, (list, tuple, np.ndarray))
            and len(pos) == 3
            and pos[0] is not False
        ):
            valid_indices.append(joint_id)
            valid_coords.append(pos)
    if len(valid_indices) < 2:
        if storage.get("line_set") is not None:
            try:
                vis.remove_geometry(storage["line_set"], reset_bounding_box=False)
            except:
                pass
            storage["line_set"] = None
        return
    lines = []
    for j1, j2 in skeleton_edges:
        if j1 in valid_indices and j2 in valid_indices:
            i1 = valid_indices.index(j1)
            i2 = valid_indices.index(j2)
            lines.append([i1, i2])
    points_np = np.array(valid_coords, dtype=np.float32)
    if storage.get("line_set") is None:
        line_set = o3d.geometry.LineSet()
        storage["line_set"] = line_set
        vis.add_geometry(line_set)
    else:
        line_set = storage["line_set"]
    line_set.points = o3d.utility.Vector3dVector(points_np)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])
    vis.update_geometry(line_set)

    # 관절 구체(spheres) 업데이트
    if storage.get("spheres") is None:
        storage["spheres"] = [None] * len(body_pos)
        storage["spheres_pos"] = [None] * len(body_pos)
    for i, pos in enumerate(body_pos):
        if (
            isinstance(pos, (list, tuple, np.ndarray))
            and len(pos) == 3
            and pos[0] is not False
        ):
            new_pos = np.array(pos, dtype=np.float32)
            if storage["spheres"][i] is None:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                sphere.paint_uniform_color([0, 0, 1])
                sphere.compute_vertex_normals()
                sphere.translate(new_pos)
                vis.add_geometry(sphere)
                storage["spheres"][i] = sphere
                storage["spheres_pos"][i] = new_pos
            else:
                prev_pos = storage["spheres_pos"][i]
                diff = new_pos - prev_pos
                storage["spheres"][i].translate(diff, relative=True)
                storage["spheres_pos"][i] = new_pos
                vis.update_geometry(storage["spheres"][i])
        else:
            if storage["spheres"] is not None and storage["spheres"][i] is not None:
                T_hide = np.eye(4)
                T_hide[2, 3] = -999999
                storage["spheres"][i].transform(T_hide)
                vis.update_geometry(storage["spheres"][i])


def draw_pointing_arrow(
    vis, body_pos, storage={"arrow": None, "prev_transform": np.eye(4)}
):
    """
    팔꿈치(관절 13)에서 시작해 손목(관절 14) 방향으로 길이 1500의 두꺼운 반직선(화살표)을 그린다.
    """
    if body_pos is None or not isinstance(body_pos, np.ndarray) or body_pos.size == 0:
        return _hide_arrow(vis, storage)
    pos_elbow = body_pos[13]
    pos_wrist = body_pos[14]
    if (False in pos_elbow) or (False in pos_wrist):
        return _hide_arrow(vis, storage)
    if storage["arrow"] is not None:
        inv_transform = np.linalg.inv(storage["prev_transform"])
        storage["arrow"].transform(inv_transform)
    else:
        arrow_length = 5000.0
        cylinder_height = arrow_length * 0.9
        cone_height = arrow_length * 0.1
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=20,
            cone_radius=0.5,
            cylinder_height=cylinder_height,
            cone_height=cone_height,
            resolution=20,
            cylinder_split=4,
            cone_split=1,
        )
        arrow.paint_uniform_color([1, 1, 0])  # 기본 색: 노란색
        arrow.compute_vertex_normals()
        vis.add_geometry(arrow)
        storage["arrow"] = arrow
        storage["prev_transform"] = np.eye(4)
    direction = pos_wrist - pos_elbow
    norm_dir = np.linalg.norm(direction)
    if norm_dir < 1e-5:
        return _hide_arrow(vis, storage)
    dir_unit = direction / norm_dir
    z_axis = np.array([0, 0, 1], dtype=float)
    v = np.cross(z_axis, dir_unit)
    c = np.dot(z_axis, dir_unit)
    s = np.linalg.norm(v)
    if s < 1e-5:
        R = np.eye(3) if c >= 0 else np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    else:
        v_n = v / s
        K = np.array([[0, -v_n[2], v_n[1]], [v_n[2], 0, -v_n[0]], [-v_n[1], v_n[0], 0]])
        R = np.eye(3) + K * s + (K @ K) * (1 - c)
    T_new = np.eye(4)
    T_new[:3, :3] = R
    T_new[:3, 3] = pos_elbow
    storage["arrow"].transform(T_new)
    storage["prev_transform"] = T_new
    vis.update_geometry(storage["arrow"])


def _hide_arrow(vis, storage):
    if storage["arrow"] is not None:
        arrow = storage["arrow"]
        inv_transform = np.linalg.inv(storage["prev_transform"])
        arrow.transform(inv_transform)
        T_hide = np.eye(4)
        T_hide[2, 3] = -999999
        arrow.transform(T_hide)
        storage["prev_transform"] = T_hide
        vis.update_geometry(arrow)


def get_arrow_segment_from_body(body_pos):
    """
    팔꿈치(관절 13)에서 시작해 손목 방향으로 길이 1500인 반직선의 시작, 끝, 및 충돌 계산용 반경(0.05)을 반환.
    """
    pos_elbow = body_pos[13]
    pos_wrist = body_pos[14]
    if (False in pos_elbow) or (False in pos_wrist):
        return None, None, 0.0
    direction = pos_wrist - pos_elbow
    norm_dir = np.linalg.norm(direction)
    if norm_dir < 1e-5:
        return None, None, 0.0
    dir_unit = direction / norm_dir
    arrow_length = 5000.0
    start = pos_elbow
    end = pos_elbow + dir_unit * arrow_length
    radius = 0.05
    return start, end, radius
