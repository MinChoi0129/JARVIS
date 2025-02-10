import open3d as o3d
from modules.point_cloud_loader import (
    load_point_cloud_from_instance_npy,
    load_point_cloud_from_txt,
)


def initialize_visualizer():
    """Open3D 시각화 초기화."""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    return vis


def setup_scene(point_cloud_file, label_file, mode, C2W):
    """포인트 클라우드 데이터 및 시각화 장면 설정."""
    # 시각화 초기화
    vis = initialize_visualizer()
    if vis is None:
        print(">>>>>>>>>>> 시각화 창 초기화에 실패했습니다.")
        return

    # 포인트 클라우드와 인스턴스 구체 로드
    # pcd, instance_spheres = load_point_cloud_from_instance_npy(
    #     point_cloud_file, label_file, mode
    # )

    ### txt ##
    pcd, instance_spheres = load_point_cloud_from_txt(point_cloud_file, mode)

    #################################### PARAMETERS ####################################
    cam_obj_scale = 300
    marker_obj_scale = 1000 / 4.7
    hand_obj_scale = 40 if mode == "space" else 20
    ###################################################################################

    # 카메라 박스 설정
    cam_box = o3d.geometry.TriangleMesh.create_box(
        width=0.5 * cam_obj_scale,
        height=(0.5 / 3) * cam_obj_scale,
        depth=(2 / 3) * cam_obj_scale,
    )
    cam_box.paint_uniform_color([0, 0, 0])
    cam_box.transform(C2W)

    # 손 구체 설정
    left_hand = o3d.geometry.TriangleMesh.create_sphere(radius=hand_obj_scale / 2)
    left_hand.paint_uniform_color([0.4, 0, 0])
    right_hand = o3d.geometry.TriangleMesh.create_sphere(radius=hand_obj_scale / 2)
    right_hand.paint_uniform_color([0, 0, 0.4])

    # 라인 세트 및 마커 설정
    line_set = o3d.geometry.LineSet()
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]])

    marker = o3d.geometry.TriangleMesh.create_box(
        width=1.0 * marker_obj_scale,
        height=1.0 * marker_obj_scale,
        depth=0.000001 * marker_obj_scale,
    )
    marker.translate([0, 0, 0])

    # 좌표 프레임 생성
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.7 * marker_obj_scale,
        origin=[0, 0, 0],
    )

    # 시각화에 추가
    vis.add_geometry(pcd)
    vis.add_geometry(cam_box)
    vis.add_geometry(left_hand)
    vis.add_geometry(right_hand)
    vis.add_geometry(marker)
    vis.add_geometry(coordinate_frame)
    vis.add_geometry(line_set)

    # 인스턴스 구체를 시각화에 추가
    for sphere in instance_spheres:
        vis.add_geometry(sphere)

    return vis, instance_spheres, left_hand, right_hand, line_set
