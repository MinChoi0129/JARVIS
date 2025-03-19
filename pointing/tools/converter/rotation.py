import open3d as o3d
import numpy as np

# 포인트 클라우드 로드
pcd = o3d.io.read_point_cloud(
    r"C:\Users\sumin\Documents\GitHub\JARVIS\pointing\data\experiment_setting_17_16_19.pcd"
)

# 이동 및 회전 스텝 정의
translation_step = 0.1  # 이동 단위
rotation_step = np.pi / 100  # 회전 단위 (약 9도)


# 이동 함수들
def move_forward(vis):
    global pcd
    pcd.translate(np.array([0, translation_step, 0]))
    vis.update_geometry(pcd)
    return False


def move_backward(vis):
    global pcd
    pcd.translate(np.array([0, -translation_step, 0]))
    vis.update_geometry(pcd)
    return False


def move_left(vis):
    global pcd
    pcd.translate(np.array([-translation_step, 0, 0]))
    vis.update_geometry(pcd)
    return False


def move_right(vis):
    global pcd
    pcd.translate(np.array([translation_step, 0, 0]))
    vis.update_geometry(pcd)
    return False


def move_up(vis):
    global pcd
    pcd.translate(np.array([0, 0, translation_step]))
    vis.update_geometry(pcd)
    return False


def move_down(vis):
    global pcd
    pcd.translate(np.array([0, 0, -translation_step]))
    vis.update_geometry(pcd)
    return False


# 회전 함수들
def rotate_x_positive(vis):
    global pcd
    R = pcd.get_rotation_matrix_from_xyz((rotation_step, 0, 0))
    pcd.rotate(R, center=(0, 0, 0))
    vis.update_geometry(pcd)
    return False


def rotate_x_negative(vis):
    global pcd
    R = pcd.get_rotation_matrix_from_xyz((-rotation_step, 0, 0))
    pcd.rotate(R, center=(0, 0, 0))
    vis.update_geometry(pcd)
    return False


def rotate_y_positive(vis):
    global pcd
    R = pcd.get_rotation_matrix_from_xyz((0, rotation_step, 0))
    pcd.rotate(R, center=(0, 0, 0))
    vis.update_geometry(pcd)
    return False


def rotate_y_negative(vis):
    global pcd
    R = pcd.get_rotation_matrix_from_xyz((0, -rotation_step, 0))
    pcd.rotate(R, center=(0, 0, 0))
    vis.update_geometry(pcd)
    return False


def rotate_z_positive(vis):
    global pcd
    R = pcd.get_rotation_matrix_from_xyz((0, 0, rotation_step))
    pcd.rotate(R, center=(0, 0, 0))
    vis.update_geometry(pcd)
    return False


def rotate_z_negative(vis):
    global pcd
    R = pcd.get_rotation_matrix_from_xyz((0, 0, -rotation_step))
    pcd.rotate(R, center=(0, 0, 0))
    vis.update_geometry(pcd)
    return False


# 저장 함수 (P키)
def save_pcd(vis):
    global pcd
    o3d.io.write_point_cloud("rotated_output.pcd", pcd)
    print("포인트 클라우드가 'rotated_output.pcd'로 저장되었습니다.")
    return False


# VisualizerWithKeyCallback 생성 및 키 바인딩
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

# 좌표축 추가 (크기와 원점은 필요에 따라 조절 가능)
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
vis.add_geometry(axes)

# 포인트 클라우드 추가
vis.add_geometry(pcd)

# 이동에 대한 키 콜백 등록
vis.register_key_callback(ord("W"), move_forward)  # 앞으로 이동
vis.register_key_callback(ord("S"), move_backward)  # 뒤로 이동
vis.register_key_callback(ord("A"), move_left)  # 왼쪽 이동
vis.register_key_callback(ord("D"), move_right)  # 오른쪽 이동
vis.register_key_callback(ord("Q"), move_up)  # 위로 이동
vis.register_key_callback(ord("E"), move_down)  # 아래로 이동

# 회전에 대한 키 콜백 등록
vis.register_key_callback(ord("I"), rotate_x_positive)  # X축 회전 (+)
vis.register_key_callback(ord("K"), rotate_x_negative)  # X축 회전 (–)
vis.register_key_callback(ord("J"), rotate_y_positive)  # Y축 회전 (+)
vis.register_key_callback(ord("L"), rotate_y_negative)  # Y축 회전 (–)
vis.register_key_callback(ord("U"), rotate_z_positive)  # Z축 회전 (+)
vis.register_key_callback(ord("O"), rotate_z_negative)  # Z축 회전 (–)

# 저장을 위한 키 콜백 등록 (P키)
vis.register_key_callback(ord("P"), save_pcd)

# 안내 메시지 출력
print("키 조작 안내:")
print("이동: W(앞), S(뒤), A(왼쪽), D(오른쪽), Q(위), E(아래)")
print("회전: I/K (X축), J/L (Y축), U/O (Z축)")
print("저장: P 키를 눌러 현재 상태의 포인트 클라우드를 저장")
print("좌표축은 창에 함께 표시됩니다.")

# 실행
vis.run()
vis.destroy_window()
