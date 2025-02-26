### visualization.py

import open3d as o3d
from modules.point_cloud_loader import load_point_cloud_from_instance_npy


def initialize_visualizer():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    return vis


def setup_scene(point_cloud_file, label_file):
    vis = initialize_visualizer()

    # (수정) load_point_cloud_from_instance_npy에서 pcd, instance_boxes, labels를 모두 받음
    pcd, instance_boxes, labels = load_point_cloud_from_instance_npy(
        point_cloud_file, label_file
    )

    # 시각화용 지오메트리 추가
    vis.add_geometry(pcd)

    cam_obj_scale = 300
    cam_box = o3d.geometry.TriangleMesh.create_box(
        width=0.5 * cam_obj_scale,
        height=(0.5 / 3) * cam_obj_scale,
        depth=(2 / 3) * cam_obj_scale,
    )
    cam_box.paint_uniform_color([0, 0, 0])
    cam_box.translate([30, 1500, 100])

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        origin=[0, 0, 0], size=300
    )
    vis.add_geometry(cam_box)
    vis.add_geometry(coordinate_frame)

    # (수정) vis, pcd, instance_boxes, labels를 함께 반환
    return vis, pcd, instance_boxes, labels
