import open3d as o3d
from modules.point_cloud_loader import load_point_cloud_from_instance_npy


def initialize_visualizer():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    return vis


def setup_scene(point_cloud_file, label_file):
    vis = initialize_visualizer()
    pcd, instance_boxes = load_point_cloud_from_instance_npy(
        point_cloud_file, label_file
    )
    cam_obj_scale = 300
    cam_box = o3d.geometry.TriangleMesh.create_box(
        width=0.5 * cam_obj_scale,
        height=(0.5 / 3) * cam_obj_scale,
        depth=(2 / 3) * cam_obj_scale,
    )
    cam_box.paint_uniform_color([0, 0, 0])
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        origin=[0, 0, 0]
    )
    vis.add_geometry(pcd)
    vis.add_geometry(cam_box)
    vis.add_geometry(coordinate_frame)
    return vis, instance_boxes
