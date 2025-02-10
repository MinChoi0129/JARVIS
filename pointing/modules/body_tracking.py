#############body_tracking.py#############
import numpy as np
import open3d as o3d
from modules.parameters import *


def get_kinect_body_positions(body_frame):
    """
    Kinect에서 스켈레톤 추적.
    관절 confidence_level이 MEDIUM(2) 이상인 관절만 반환.
    일부 관절이 추적 안 되더라도 오류 없이 동작.
    """
    # body_frame이 None이거나, 몸체가 하나도 감지되지 않았으면 False
    if not body_frame:
        return False, None
    if body_frame.get_num_bodies() == 0:
        return False, None

    try:
        skeleton = body_frame.get_body_skeleton(0)
    except SystemExit:
        return False, None
    except Exception as e:
        raise Exception("Skeleton Fetching Error")

    valid_joints_3d = []
    for joint_id in range(32):
        joint = skeleton.joints[joint_id]
        if joint.confidence_level >= 1:  # 0=NONE,1=LOW,2=MEDIUM,3=HIGH
            pos = joint.position.xyz
            # Open3D 상에서 z축이 위쪽, x축이 오른쪽, y축이 전/후방이 되도록 변환 예시
            # (기존: Kinect -> x, y, z / 보정: x, z, -y)
            valid_joints_3d.append([pos.x, pos.z, -pos.y])
        else:
            valid_joints_3d.append([False, False, False])

    # 유효 관절이 하나도 없으면 False 처리
    if not valid_joints_3d:
        return False, None

    joints_3d_arr = np.array(valid_joints_3d, dtype=np.float32)
    return True, joints_3d_arr.view()


def update_body_positions(vis, body_pos, storage={"line_set": None}):
    """
    body_pos:
      - None, 혹은
      - numpy 배열(길이 32), 각 관절마다 [x, y, z] 또는 False가 들어 있음
    vis: Open3D Visualizer 객체
    storage: LineSet 객체 등을 캐싱하기 위한 dict
    """

    # 1) body_pos가 None이거나, numpy array이면서 size=0이면 -> 추적 실패 상황
    if body_pos is None:
        # 이미 등록된 line_set 제거
        if storage["line_set"] is not None:
            try:
                vis.remove_geometry(storage["line_set"], reset_bounding_box=False)
            except:
                pass
            storage["line_set"] = None
        return vis

    # 만약 body_pos가 numpy 배열이고, 그 길이가 0이면 마찬가지로 처리
    if isinstance(body_pos, np.ndarray) and body_pos.size == 0:
        if storage["line_set"] is not None:
            try:
                vis.remove_geometry(storage["line_set"], reset_bounding_box=False)
            except:
                pass
            storage["line_set"] = None
        return vis

    # 2) body_pos가 numpy 배열(또는 list)인 경우 처리
    #    - 각 관절 마다 False or [x,y,z] 형태라고 가정
    valid_indices = []
    valid_coords = []

    # body_pos가 ndarray면 enumerate에 바로 넣어도 되지만,
    # 리스트로 가정해도 동일
    for joint_id, pos in enumerate(body_pos):
        # 만약 pos가 False라면, 관절 미추적 상태
        if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) == 3:
            valid_indices.append(joint_id)
            valid_coords.append(pos)
        else:
            # pos가 False이거나, 엉뚱한 형태 => skip
            pass

    # 유효 관절 2개 미만 -> 라인도 그릴 수 없음 -> 기존 라인셋 제거
    if len(valid_indices) < 2:
        if storage["line_set"] is not None:
            try:
                vis.remove_geometry(storage["line_set"], reset_bounding_box=False)
            except:
                pass
            storage["line_set"] = None
        return vis

    # 3) skeleton_edges에서 양 끝이 모두 유효 관절인 라인만 연결
    lines = []
    for j1, j2 in skeleton_edges:
        if j1 in valid_indices and j2 in valid_indices:
            i1 = valid_indices.index(j1)
            i2 = valid_indices.index(j2)
            lines.append([i1, i2])

    points_np = np.array(valid_coords, dtype=np.float32)

    # 4) LineSet 객체 생성/업데이트
    if storage["line_set"] is None:
        line_set = o3d.geometry.LineSet()
        storage["line_set"] = line_set
        vis.add_geometry(line_set)
    else:
        line_set = storage["line_set"]

    line_set.points = o3d.utility.Vector3dVector(points_np)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])

    # 5) 뷰어 업데이트
    vis.update_geometry(line_set)

    return vis


def draw_pointing_cylinder(
    vis, body_pos, storage={"cylinder": None, "prev_transform": np.eye(4)}
):
    """
    관절 13 (ElbowRight) ~ 14 (WristRight) 벡터를 따라
    고정 길이 실린더를 회전, 위치만 업데이트한다.

    1) 프로그램 실행 시(또는 처음 유효 관절 발견 시) 딱 한 번 cylinder 생성
    2) 매 프레임: 기존 transform 되돌리고, 새 transform만 적용
    3) 비균등 스케일은 사용하지 않음 -> height는 고정
    """
    # 1) body_pos가 유효한지 검사
    if body_pos is None or not isinstance(body_pos, np.ndarray) or body_pos.size == 0:
        return _hide_cylinder(vis, storage)

    pos13 = body_pos[13]  # K4ABT_JOINT_ELBOW_RIGHT
    pos14 = body_pos[14]  # K4ABT_JOINT_WRIST_RIGHT

    # 추적 실패 시 -> 숨기기
    if (False in pos13) or (False in pos14):
        return _hide_cylinder(vis, storage)

    # 2) 기존 transform을 되돌리기
    if storage["cylinder"] is not None:
        inv_transform = np.linalg.inv(storage["prev_transform"])
        storage["cylinder"].transform(inv_transform)
    else:
        # 처음으로 cylinder를 생성(단 한 번만)
        # 예: radius=0.03, height=2.0 으로 고정
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=30, height=3000, resolution=20
        )
        cylinder.paint_uniform_color([1, 1, 0])  # 노란색
        cylinder.compute_vertex_normals()
        vis.add_geometry(cylinder)

        storage["cylinder"] = cylinder
        storage["prev_transform"] = np.eye(4)

    cylinder_mesh = storage["cylinder"]

    # 3) Elbow->Wrist 방향 벡터
    direction = pos14 - pos13

    if np.linalg.norm(direction) < 1e-5:
        # 관절이 거의 겹치면 숨김
        return _hide_cylinder(vis, storage)

    # 4) 회전행렬 계산
    #    cylinder의 로컬 z축([0,0,1])을 direction 방향과 일치시키기
    z_axis = np.array([0, 0, 1], dtype=float)
    dir_unit = direction / np.linalg.norm(direction)

    v = np.cross(z_axis, dir_unit)  # 회전축
    c = np.dot(z_axis, dir_unit)  # cos(θ)
    s = np.linalg.norm(v)  # sin(θ)

    if s < 1e-5:
        # 거의 평행
        R = np.eye(3)
        if c < 0:
            # 정반대 방향
            R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    else:
        v_n = v / s
        K = np.array([[0, -v_n[2], v_n[1]], [v_n[2], 0, -v_n[0]], [-v_n[1], v_n[0], 0]])
        R = np.eye(3) + K * s + (K @ K) * (1 - c)

    # 5) 평행이동
    #    - create_cylinder(height=2.0)은 로컬 좌표에서 z=0 ~ z=2 범위
    #    - z=0 위치를 elbow(pos13)에 놓으면, 그 끝(z=2)은 elbow + dir_unit*2 쪽으로 뻗음
    #    - 즉, Wrist보다 훨씬 멀리(2m)까지 이어진다는 의미
    T_new = np.eye(4)
    T_new[:3, :3] = R
    T_new[:3, 3] = pos13  # base를 Elbow 위치에 둠

    # 6) 실린더에 변환 적용
    cylinder_mesh.transform(T_new)

    # 7) 다음 프레임을 위해 prev_transform 저장
    storage["prev_transform"] = T_new

    vis.update_geometry(cylinder_mesh)
    return vis


def _hide_cylinder(vis, storage):
    """
    스켈레톤이 유효하지 않을 때,
    이미 생성된 cylinder가 있다면 시야 밖으로 보낸다.
    (remove_geometry 대신, 메모리에 계속 보유)
    """
    if storage["cylinder"] is not None:
        cylinder_mesh = storage["cylinder"]
        # 1) 이전 transform 되돌리기
        inv_transform = np.linalg.inv(storage["prev_transform"])
        cylinder_mesh.transform(inv_transform)

        # 2) 시야 밖으로 이동 (예: z=-9999)
        T_hide = np.eye(4)
        T_hide[2, 3] = -999999
        cylinder_mesh.transform(T_hide)

        storage["prev_transform"] = T_hide

        vis.update_geometry(cylinder_mesh)

    return vis


def get_cylinder_segment_from_body(body_pos):
    """
    예시 함수:
    ElbowRight(13)~WristRight(14)를 이용해
    start, end, radius를 계산해 리턴.
    """
    pos13 = body_pos[13]
    pos14 = body_pos[14]
    if (False in pos13) or (False in pos14):
        # 추적 안 되는 경우
        return None, None, 0.0

    direction = pos14 - pos13
    length = np.linalg.norm(direction)
    if length < 1e-5:
        return None, None, 0.0

    # 원하는 길이로 (ex) 2.0 m 고정
    dir_unit = direction / length
    start = pos13
    end = pos13 + dir_unit * 2.0
    radius = 0.03

    return start, end, radius
