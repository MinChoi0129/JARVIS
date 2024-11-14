import cv2
from modules.kinect_manager import initialize_kinect, reset_kinect_devices
from modules.visualization import setup_scene
from modules.transformation import get_transformation_matrix
from modules.hand_tracking import get_kinect_hand_positions, update_hand_positions
from modules.parameters import *


def logger(*data):
    if len(data) == 1:  # 일반 출력
        print(data[0])
    else:  # 예외 처리
        e, msg = data
        print("=======================================")
        print(e)
        print("=======================================")
        print(msg)


def main():
    try:
        # Kinect 초기화
        try:
            reset_kinect_devices()
            device, body_tracker = initialize_kinect()
        except Exception as e:
            logger(e, ">>>>>>>>>>> Kinect 초기화 오류:")
            return

        # Transformation Matrix 계산
        try:
            C2W = get_transformation_matrix(
                device,
                aruco_dict,
                aruco_params,
                marker_length,
                camera_matrix,
                dist_coeffs,
            )
        except Exception as e:
            logger(e, f">>>>>>>>>>> Transformation Matrix 계산 오류")
            return

        # 포인트 클라우드 및 장면 설정
        try:
            vis, instance_spheres, left_hand, right_hand, line_set = setup_scene(
                point_cloud_file="data/702_coord.npy",  # "data/702_coord.npy"
                label_file="data/702_instance_with_class_pred.npy",
                mode="object",
                C2W=C2W,
            )
        except FileNotFoundError as e:
            logger(e, f">>>>>>>>>>> 파일을 찾을 수 없습니다")
            return
        except Exception as e:
            logger(e, f">>>>>>>>>>> 포인트 클라우드 로드 오류")
            return

        ############################### 메인 루프 ###############################
        while True:
            capture = device.update()
            ret, frame = capture.get_color_image()
            if not ret:
                logger(">>>>>>>>>>> Kinect 프레임을 가져올 수 없습니다.")
                continue

            # 손 위치 업데이트
            try:
                body_frame = body_tracker.update()
                ret, left_hand_pos, right_hand_pos = get_kinect_hand_positions(
                    body_frame, C2W
                )

                if ret:
                    vis, distance_cm = update_hand_positions(
                        vis,
                        left_hand,
                        right_hand,
                        line_set,
                        left_hand_pos,
                        right_hand_pos,
                        instance_spheres,
                    )
                    # print(f"거리: {distance_cm} cm")

            except Exception as e:
                logger(
                    e, f">>>>>>>>>>> Skeleton에서 손 위치를 가져오는데 실패했습니다."
                )
                continue

            vis.poll_events()
            vis.update_renderer()

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        device.close()
        cv2.destroyAllWindows()
        vis.destroy_window()

    except Exception as e:
        logger(e, f">>>>>>>>>>> while 문 예외 발생")

    finally:  # 장치 초기화 해제 및 창 닫기
        cv2.destroyAllWindows()

        variables = {**locals(), **locals()}
        if "device" in variables:
            device.close()
        if "vis" in variables:
            vis.destroy_window()

        logger(">>>>>>>>>>> 정상적으로 종료되었습니다.")


# main 함수 실행
if __name__ == "__main__":
    main()
