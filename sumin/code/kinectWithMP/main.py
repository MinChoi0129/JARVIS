import cv2
import time
from depth_processing import process_depth_information
from kinect_capture import initialize_kinect, get_kinect_capture, close_kinect
from mp_hand_tracking_model import process_hand_landmarks
from pcd_on_markerWorld import detect_aruco_marker, load_point_cloud_from_txt, display_point_cloud

point_cloud_file = '../chair_20.txt'


def main():
    # Kinect 초기화
    try:
        kinect = initialize_kinect()
        if kinect is None:
            print("Error: Kinect could not be initialized.")
            return
    except Exception as e:
        print(f"Error initializing Kinect: {e}")
        return

    # 포인트 클라우드 파일 로드
    try:
        pcd = load_point_cloud_from_txt(point_cloud_file)
        print("Point cloud loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Point cloud file '{point_cloud_file}' not found.")
        return
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return

    try:
        while True:
            try:
                color_capture, depth_capture = get_kinect_capture(kinect)
                if color_capture is None or depth_capture is None:
                    print("Warning: No image data received from Kinect.")
                    continue

                color_image = color_capture[1]
                depth_image = depth_capture[1]

                processed_image, hand_results = process_hand_landmarks(
                    color_image)
                if hand_results:
                    hand_data = process_depth_information(hand_results.multi_hand_landmarks[0],
                                                          depth_image,
                                                          processed_image.shape[1],
                                                          processed_image.shape[0])
                    print("Hand data:", hand_data[0])

                # 결과를 화면에 표시
                cv2.imshow("Processed Image", processed_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # 포인트 클라우드
                rvec, tvec = detect_aruco_marker(color_image)
                if rvec is not None and tvec is not None:
                    display_point_cloud(pcd, rvec, tvec)

            except Exception as e:
                print(f"Error in processing loop: {e}")
                continue

    except KeyboardInterrupt:
        print("Process interrupted by user.")

    finally:
        close_kinect(kinect)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
