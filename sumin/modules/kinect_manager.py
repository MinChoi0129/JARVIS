import os
import ctypes
import pykinect_azure as pykinect


def reset_kinect_devices():
    """Azure Kinect 장치를 강제로 초기화 및 닫기."""
    if os.name == "nt":
        try:
            k4a = ctypes.windll.LoadLibrary(
                r"C:\Program Files\Azure Kinect SDK v1.4.1\sdk\windows-desktop\amd64\release\bin\k4a.dll"
            )
            k4a.k4a_device_close.restype = ctypes.c_void_p
            k4a.k4a_device_close.argtypes = [ctypes.c_void_p]
            handle = ctypes.c_void_p(0)
            k4a.k4a_device_close(handle)
        except Exception as e:
            print(">>>>>>>>>>> Kinect 장치 재설정 오류:", e)
    else:
        print(">>>>>>>>>>> 이 기능은 Windows 환경에서만 지원됩니다.")


def initialize_kinect():
    """Kinect 장치 및 트래커 초기화."""
    # reset_kinect_devices()
    pykinect.initialize_libraries(track_body=True)
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    device = pykinect.start_device(config=device_config)
    body_tracker = pykinect.start_body_tracker()
    return device, body_tracker
