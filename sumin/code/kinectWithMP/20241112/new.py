import cv2
import numpy as np
import open3d as o3d
import pykinect_azure as pykinect
from parameters import *


# Kinect 초기화
pykinect.initialize_libraries(track_body=True)
device_config = pykinect.default_configuration
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
device = pykinect.start_device(config=device_config)
bodyTracker = pykinect.start_body_tracker()

# 모드 설정
mode = "object"

vis = o3d.visualization.Visualizer()
vis.create_window()


def load_point_cloud_from_txt(file_path, mode):
    point_cloud_data = np.loadtxt(file_path)

    scale = 100 if mode == "object" else 1000
    points = point_cloud_data[:, :3] * scale
    colors = point_cloud_data[:, 3:] / 255.0

    center = np.mean(points, axis=0)
    points -= center
    points += np.array([84, 84, 168])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def get_kinect_hand_positions(body_frame, c2w):
    if body_frame:
        skeleton = body_frame.get_body_skeleton()

        lhp = skeleton.joints[8].position.xyz
        rhp = skeleton.joints[15].position.xyz

        left_hand_xyz = np.array([lhp.x + 20, lhp.y, lhp.z, 1]).T
        right_hand_xyz = np.array([rhp.x + 20, rhp.y, rhp.z, 1]).T

        return (c2w @ left_hand_xyz)[:3], (c2w @ right_hand_xyz)[:3]
    else:
        return None, None


def load_pcd_and_initialize_visualizer(point_cloud_file, mode):
    pcd = load_point_cloud_from_txt(point_cloud_file, mode)

    cam_obj_scale = 300
    marker_obj_scale = 1000 / 4.7
    hand_obj_scale = 40 if mode == "space" else 20

    cam_box = o3d.geometry.TriangleMesh.create_box(
        width=0.5 * cam_obj_scale,
        height=(0.5 / 3) * cam_obj_scale,
        depth=(2 / 3) * cam_obj_scale,
    )
    cam_box.paint_uniform_color([0, 0, 0])

    marker = o3d.geometry.TriangleMesh.create_box(
        width=1.0 * marker_obj_scale,
        height=1.0 * marker_obj_scale,
        depth=0.000001 * marker_obj_scale,
    )
    marker.translate([0, 0, 0])

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.7 * marker_obj_scale,
        origin=[0, 0, 0],
    )

    left_hand = o3d.geometry.TriangleMesh.create_sphere(radius=hand_obj_scale / 2)
    left_hand.paint_uniform_color([1, 0, 0])

    right_hand = o3d.geometry.TriangleMesh.create_sphere(radius=hand_obj_scale / 2)
    right_hand.paint_uniform_color([0, 0, 1])

    vis.add_geometry(pcd)
    vis.add_geometry(cam_box)
    vis.add_geometry(coordinate_frame)
    vis.add_geometry(marker)
    vis.add_geometry(left_hand)
    vis.add_geometry(right_hand)

    return cam_box, marker, coordinate_frame, left_hand, right_hand


def getTransformationMatrix():
    while True:
        capture = device.update()
        ret, frame = capture.get_color_image()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=aruco_params
        )

        if ids is None or len(ids) == 0:
            print("Marker Not Found!")
            continue

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[0],
            markerLength=marker_length,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
        )

        rvec, tvec = rvec[0], tvec[0]
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        C2W = np.linalg.inv(T)
        return C2W


try:
    # 포인트 클라우드 파일 경로 설정
    point_cloud_file = "702.txt"

    cam_box, marker, coordinate_frame, left_hand, right_hand = (
        load_pcd_and_initialize_visualizer(point_cloud_file, mode)
    )

    C2W = getTransformationMatrix()
    cam_box.transform(C2W)

    # 두 손 사이의 선을 그릴 LineSet 객체 생성
    line_set = o3d.geometry.LineSet()
    vis.add_geometry(line_set)  # 선을 시각화에 추가
    vis.add_geometry(left_hand)
    vis.add_geometry(right_hand)

    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]])

    while True:
        capture = device.update()
        ret, frame = capture.get_color_image()
        if not ret:
            continue

        try:
            body_frame = bodyTracker.update()
            left_hand_pos, right_hand_pos = get_kinect_hand_positions(body_frame, C2W)
        except:
            continue

        left_hand.translate((left_hand_pos - np.asarray(left_hand.get_center())))
        right_hand.translate((right_hand_pos - np.asarray(right_hand.get_center())))

        left_hand_xyz = np.asarray(left_hand.get_center())
        right_hand_xyz = np.asarray(right_hand.get_center())
        distance = np.linalg.norm(left_hand_xyz - right_hand_xyz)

        # 선을 업데이트
        line_set.points = o3d.utility.Vector3dVector([left_hand_xyz, right_hand_xyz])

        print(distance)

        vis.update_geometry(left_hand)
        vis.update_geometry(right_hand)
        vis.update_geometry(line_set)

        vis.poll_events()
        vis.update_renderer()

        cv2.imshow("Aruco Marker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    vis.destroy_window()

except Exception as e:
    print(f"예외 발생: {e}")

finally:
    cv2.destroyAllWindows()
    vis.destroy_window()
    device.close()
