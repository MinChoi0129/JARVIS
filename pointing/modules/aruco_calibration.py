import cv2
import numpy as np
from modules.parameters import *


def get_transformation_matrix(
    device, aruco_dict, aruco_params, marker_length, camera_matrix, dist_coeffs
):
    """Transformation Matrix 계산."""
    print(">>>>>>>>>>> Marker 탐색을 시작합니다.!")

    while True:
        capture = device.update()
        ret, frame = capture.get_color_image()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=aruco_params
        )

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        if ids is None or len(ids) == 0:
            continue

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[0], marker_length, camera_matrix, dist_coeffs
        )
        R, _ = cv2.Rodrigues(rvec[0])
        W2C = np.eye(4)
        W2C[:3, :3] = R
        W2C[:3, 3] = tvec[0].flatten()
        print(f">>>>>>>>>>> 마커를 찾았습니다! {rvec.flatten(), tvec.flatten()}")

        C2W = np.eye(4)
        C2W[:3, 3] = np.array([0, 1500, 100])

        return C2W @ np.linalg.inv(W2C)
