import cv2
import numpy as np
import open3d as o3d
import pykinect_azure as pykinect


fx = 624.182229
fy = 624.182229
cx = 640.000000
cy = 360.000000
k1 = 0.083126
k2 = -0.029291
p1 = -0.001305
p2 = -0.002851

marker_length = 5


# Aruco 마커 설정
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()

# 카메라 행렬과 왜곡 계수 (캘리브레이션 필요)
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
dist_coeffs = np.array([k1, k2, p1, p2])


# kinect
pykinect.initialize_libraries(track_body=True)

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
# print(device_config)

# Start device
device = pykinect.start_device(config=device_config)

# Start body tracker
bodyTracker = pykinect.start_body_tracker()
#################

# Aruco 마커를 인식하고 월z드 좌표계 정의


def detect_aruco_marker(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs)
        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                              rvecs[i], tvecs[i], marker_length)
            return rvecs[i], tvecs[i]  # 회전 및 이동 벡터
    return None, None

# 포인트 클라우드를 파일에서 읽기


def load_point_cloud_from_txt(file_path, scale=1.0, y_offset=0.1):
    point_cloud_data = np.loadtxt(file_path)

    points = point_cloud_data[:, :3]  # x, y, z 좌표
    colors = point_cloud_data[:, 3:] / 255.0  # R, G, B 값을 [0, 1] 범위로 변환

    center = np.mean(points, axis=0)
    points -= center
    points[:, 1] += y_offset  # Y축 방향으로 이동
    points *= scale  # 스케일 적용

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def kinect():
    # Get body tracker frame
    body_frame = bodyTracker.update()

    a = body_frame.get_body()
    b = body_frame.get_body_skeleton()
    c = body_frame.get_body_index_map()

    lhp = b.joints[8].position.xyz
    rhp = b.joints[15].position.xyz

    left_hand_xyz = lhp.x, lhp.y, lhp.z
    right_hand_xyz = rhp.x, rhp.y, rhp.z

    return left_hand_xyz, right_hand_xyz


def display_point_cloud(pcd, rvec, tvec, box_scale=1.0):
    # 마커의 회전 및 이동 변환을 적용하여 포인트 클라우드를 마커 위치에 맞춤
    R, _ = cv2.Rodrigues(rvec)  # 회전 벡터를 회전 행렬로 변환
    W2K = np.eye(4)
    W2K[:3, :3] = R
    W2K[:3, 3] = tvec.flatten()

    K2W = np.linalg.inv(W2K)

    # box_scale을 적용하여 박스의 크기 조정
    cam_box = o3d.geometry.TriangleMesh.create_box(
        width=1.0 * box_scale, height=0.1 * box_scale, depth=1.0 * box_scale)

    cam_box.translate([-0.5 * box_scale, -0.05 *
                       box_scale, -0.5 * box_scale])  # 중심 조정
    cam_box.paint_uniform_color([0, 0, 0])  # 박스 색을 회색으로 설정

    cam_box.transform(K2W)
###########################
    marker = o3d.geometry.TriangleMesh.create_box(
        width=1.0 * box_scale, height=0.1 * box_scale, depth=1.0 * box_scale)
    marker.translate([0, 0, 0])
############################

    right_hand = o3d.geometry.TriangleMesh.create_sphere()

    left_hand = o3d.geometry.TriangleMesh.create_sphere()

######################
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.7 * box_scale, origin=[0, 0, 0])

    vis = o3d.visualization.draw_geometries(
        [pcd, cam_box, coordinate_frame, marker,
            right_hand, left_hand], zoom=0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0475, 1.532],
        up=[-0.0694, -0.9768, 0.2024])

    while kinect:
        l, r = kinect()
        left_hand.translate = (l)
        right_hand.translate = (r)
        vis.update_geometry(left_hand)
        vis.update_geometry(right_hand)


# 예제 메인 코드
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    # 텍스트 파일에서 포인트 클라우드 로드
    # 여기 파일 경로를 입력하세요
    point_cloud_file = '../pcd_data/Area_1/conferenceRoom_2/Annotations/chair_20.txt'
    # 포인트 클라우드 로드 시 스케일을 2로 설정
    pcd = load_point_cloud_from_txt(
        point_cloud_file, scale=30.0, y_offset=-0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rvec, tvec = detect_aruco_marker(frame)

        if rvec is not None and tvec is not None:
            display_point_cloud(pcd, rvec, tvec, box_scale=10.0)

        cv2.imshow('Aruco Marker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
