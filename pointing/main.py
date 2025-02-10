import cv2, time
from modules.parameters import *
from modules.kinect_manager import initialize_kinect, reset_kinect_devices
from modules.visualization import setup_scene
from modules.body_tracking import (
    get_kinect_body_positions,
    update_body_positions,
    draw_pointing_cylinder,
    get_cylinder_segment_from_body,
)
from modules.bounding_box_collision import (
    check_cylinder_hit_instances,
    create_aabb_lineset,
)


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

        # 포인트 클라우드 및 장면 설정
        try:
            vis, instance_boxes = setup_scene(
                point_cloud_file="data/702_coord.npy",  # "data/702_coord.npy"
                label_file="data/702_instance_with_class_pred.npy",
            )
        except FileNotFoundError as e:
            logger(e, f">>>>>>>>>>> 파일을 찾을 수 없습니다")
            return
        except Exception as e:
            logger(e, f">>>>>>>>>>> 포인트 클라우드 로드 오류")
            return

            # draw_pointing_cylinder() 함수에서 사용할 스토리지
        cylinder_storage = {
            "cylinder": None,
            "prev_transform": np.eye(4),
        }

        time.sleep(2)

        # AABB를 시각화 (LineSet)으로 표시
        line_sets = []
        for box_info in instance_boxes:
            aabb = box_info["aabb"]
            ls = create_aabb_lineset(aabb, color=[0, 1, 0])  # 예: 녹색
            vis.add_geometry(ls)
            line_sets.append(ls)

        while True:
            capture = device.update()
            ret, frame = capture.get_color_image()
            if not ret:
                logger(">>>>>>>>>>> Kinect 프레임을 가져올 수 없습니다.")
                continue

            try:
                body_frame = body_tracker.update()
                ret_body, body_pos = get_kinect_body_positions(body_frame)

                if ret_body:
                    update_body_positions(vis, body_pos)
                    draw_pointing_cylinder(vis, body_pos, cylinder_storage)
                    start, end, cyl_radius = get_cylinder_segment_from_body(body_pos)

                    try:
                        hit_labels = check_cylinder_hit_instances(
                            start=start,
                            end=end,
                            cylinder_radius=cyl_radius,
                            instance_boxes=instance_boxes,
                        )

                        if hit_labels:
                            print(f"Pointing cylinder hits: {hit_labels}")
                        else:
                            pass

                    except Exception as e:
                        logger(
                            e,
                            f">>>>>>>>>>> 충돌 감지에 실패했습니다.",
                        )
                        continue

                else:
                    # 스켈레톤 자체 추적 실패 -> 실린더 숨김
                    draw_pointing_cylinder(vis, None, cylinder_storage)

            except Exception as e:
                logger(
                    e, f">>>>>>>>>>> Skeleton에서 body pos를 가져오는 데 실패했습니다."
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
        # if "vis" in variables:
        #     vis.destroy_window()

        logger(">>>>>>>>>>> 정상적으로 종료되었습니다.")


# main 함수 실행
if __name__ == "__main__":
    main()
