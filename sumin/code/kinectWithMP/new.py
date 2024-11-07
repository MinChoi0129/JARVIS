import cv2
import numpy as np
import open3d as o3d
import pykinect_azure as pykinect

# 카메라 내부 파라미터 설정
fx = 608.427060
fy = 608.427060
cx = 640.000000
cy = 360.000000
k1 = 0.077901
k2 = -0.053721
p1 = -0.001677
p2 = -0.001842


marker_length = 5

# ArUco 마커 설정
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()

# 카메라 행렬과 왜곡 계수
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
dist_coeffs = np.array([k1, k2, p1, p2, 0])

# Kinect 초기화
pykinect.initialize_libraries(track_body=True)
device_config = pykinect.default_configuration
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
device = pykinect.start_device(config=device_config)
bodyTracker = pykinect.start_body_tracker()

vis = o3d.visualization.Visualizer()
vis.create_window()


def detect_aruco_marker(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=aruco_params)

    if ids is not None and len(ids) > 0:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs)
        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                              rvecs[i], tvecs[i], marker_length)
            return rvecs[i], tvecs[i]
    return None, None


def load_point_cloud_from_txt(file_path, scale=1.0, x_offset=0.0, y_offset=0.0, z_offset=0.0):
    point_cloud_data = np.loadtxt(file_path)

    points = point_cloud_data[:, :3]
    colors = point_cloud_data[:, 3:] / 255.0

    center = np.mean(points, axis=0)
    points -= center
    points[:, 0] += x_offset
    points[:, 1] += y_offset
    points[:, 2] += z_offset
    points *= scale

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def get_kinect_hand_positions(body_frame, c2w):
    if body_frame:
        skeleton = body_frame.get_body_skeleton()

        lhp = skeleton.joints[8].position.xyz
        rhp = skeleton.joints[15].position.xyz

        left_hand_xyz = np.array([lhp.x, lhp.y, lhp.z, 1]).T
        right_hand_xyz = np.array([rhp.x, rhp.y, rhp.z, 1]).T

        return (c2w@left_hand_xyz)[:3], (c2w@right_hand_xyz)[:3]
    else:
        return None, None


if __name__ == "__main__":
    try:
        # 포인트 클라우드 파일 경로 설정
        point_cloud_file = 'chair_20.txt'
        pcd = load_point_cloud_from_txt(
            point_cloud_file, scale=20.0, z_offset=1)

        box_scale = 10.0

        cam_box = o3d.geometry.TriangleMesh.create_box(
            width=0.5 * box_scale, height=0.25 * box_scale, depth=1.5 * box_scale)
        cam_box.translate([-0.5 * box_scale, -0.05 *
                           box_scale, -0.5 * box_scale])
        cam_box.paint_uniform_color([0, 0, 0])

        marker = o3d.geometry.TriangleMesh.create_box(
            width=1.0 * box_scale, height=1.0 * box_scale, depth=0.1 * box_scale)
        marker.translate([0, 0, 0])

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.7 * box_scale, origin=[0, 0, 0])

        left_hand = o3d.geometry.TriangleMesh.create_sphere(radius=1)
        left_hand.paint_uniform_color([1, 0, 0])
        left_hand.translate([0, 0, 0])
        right_hand = o3d.geometry.TriangleMesh.create_sphere(radius=1)
        right_hand.paint_uniform_color([0, 0, 1])
        right_hand.translate([0, 0, 0])

        vis.add_geometry(pcd)
        vis.add_geometry(cam_box)
        vis.add_geometry(coordinate_frame)
        vis.add_geometry(marker)
        vis.add_geometry(left_hand)
        vis.add_geometry(right_hand)

    #######################
        capture = device.update()
        ret, frame = capture.get_color_image()

        # OpenCV에서 사용할 수 있도록 이미지 형식 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = cv2.resize(frame, (1280, 720))

        rvec, tvec = detect_aruco_marker(frame)

        if rvec is not None and tvec is not None:
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()
            C2W = np.linalg.inv(T)

            cam_box.transform(C2W)
        else:
            print("Marker Not Found!")
            exit(0)

        while True:
            # Kinect에서 캡처 가져오기
            capture = device.update()

            # 컬러 이미지 가져오기
            ret, frame = capture.get_color_image()
            if not ret:
                print(213)
                continue

            # 바디 트래킹 업데이트
            try:
                body_frame = bodyTracker.update()
            except:
                continue

            try:
                left_hand_pos, right_hand_pos = get_kinect_hand_positions(
                    body_frame, C2W)
                print(left_hand_pos, right_hand_pos)
            except:
                pass

            left_hand.translate(
                (left_hand_pos / 50 - np.asarray(left_hand.get_center())))
            right_hand.translate(
                (right_hand_pos / 50 - np.asarray(right_hand.get_center())))

            vis.update_geometry(left_hand)
            vis.update_geometry(right_hand)

            vis.poll_events()
            vis.update_renderer()

            cv2.imshow('Aruco Marker', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        vis.destroy_window()

    ##################
    except:
        pass
    finally:
        cv2.destroyAllWindows()
        vis.destroy_window()
        device.close()
