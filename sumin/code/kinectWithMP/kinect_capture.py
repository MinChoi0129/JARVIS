from pyKinectAzure import pykinect_azure as pykinect
import cv2


def initialize_kinect():
    # Kinect SDK 및 Body Tracking SDK 경로 설정
    pykinect.initialize_libraries(
        #     "C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin",
        #     "C:\\Program Files\\Azure Kinect Body Tracking SDK\\tools"
    )

    # Kinect 디바이스 초기화 및 시작
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED

    kinect = pykinect.start_device(config=device_config)
    return kinect


def get_kinect_capture(kinect):
    # 이미지 캡처 및 반환
    capture = kinect.update()
    color_image = capture.get_color_image()
    depth_image = capture.get_depth_image()
    return color_image, depth_image


def close_kinect(kinect):
    # Kinect 종료
    kinect.stop_cameras()
