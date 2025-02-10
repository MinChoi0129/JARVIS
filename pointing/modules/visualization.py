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


def setup_scene(point_cloud_file, label_file):
    """포인트 클라우드 데이터 및 시각화 장면 설정."""
    # 시각화 초기화
    vis = initialize_visualizer()
    if vis is None:
        print(">>>>>>>>>>> 시각화 창 초기화에 실패했습니다.")
        return

    # 포인트 클라우드와 인스턴스 구체 로드
    pcd, instance_boxes = load_point_cloud_from_instance_npy(
        point_cloud_file, label_file
    )

    ### txt ##
    # pcd, instance_spheres = load_point_cloud_from_txt(point_cloud_file)

    #################################### PARAMETERS ####################################
    cam_obj_scale = 300
    ###################################################################################

    # 카메라 박스 설정
    cam_box = o3d.geometry.TriangleMesh.create_box(
        width=0.5 * cam_obj_scale,
        height=(0.5 / 3) * cam_obj_scale,
        depth=(2 / 3) * cam_obj_scale,
    )
    cam_box.paint_uniform_color([0, 0, 0])

    # 좌표 프레임 생성
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        origin=[0, 0, 0],
    )

    # 시각화에 추가
    vis.add_geometry(pcd)
    vis.add_geometry(cam_box)
    vis.add_geometry(coordinate_frame)

    return vis, instance_boxes
