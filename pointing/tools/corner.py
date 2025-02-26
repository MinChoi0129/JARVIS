import cv2
import numpy as np


def main():
    # 6x6 ArUco dictionary 생성 (최대 250개 마커)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()

    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print("마커 검출 시작 (q를 누르면 종료)")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ArUco 마커 검출
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=aruco_params
        )

        if ids is not None:
            # 검출된 마커들을 이미지에 그립니다.
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            # 각 마커에 대해...
            for i, marker_id in enumerate(ids.flatten()):
                # id 0인 마커만 선택
                if marker_id == 0:
                    marker_corners = corners[i].reshape((4, 2))
                    for idx, (x, y) in enumerate(marker_corners):
                        # 코너 위치에 원을 그리고 인덱스 번호 표시
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                        cv2.putText(
                            frame,
                            str(idx),
                            (int(x) + 10, int(y) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )
                        print(f"마커 id 0, 코너 {idx}: ({x:.2f}, {y:.2f})")

        cv2.imshow("Marker Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
