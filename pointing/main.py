import cv2
import time
import numpy as np
from modules.parameters import *
from modules.kinect_manager import initialize_kinect, reset_kinect_devices
from modules.visualization import setup_scene
from modules.body_tracking import (
    get_kinect_body_positions,
    update_body_positions,
    draw_pointing_arrow,
    get_arrow_segment_from_body,
)
from modules.bounding_box_collision import (
    check_cylinder_hit_instances,
    create_aabb_lineset,
)


def main():
    try:
        # Kinect 초기화
        try:
            reset_kinect_devices()
            device, body_tracker = initialize_kinect()
        except Exception as e:
            print("Kinect 초기화 오류:", e)
            return

        # 포인트 클라우드 및 시각화 장면 설정
        try:
            vis, instance_boxes = setup_scene(
                point_cloud_file="data/702_coord.npy",
                label_file="data/702_instance_with_class_pred.npy",
            )
        except Exception as e:
            print("포인트 클라우드 로드 오류:", e)
            return

        # 각 인스턴스의 바운딩박스를 LineSet으로 생성 및 저장 (인스턴스 색상을 사용)
        for inst in instance_boxes:
            ls = create_aabb_lineset(inst)  # inst 내 "color"를 사용
            vis.add_geometry(ls)
            inst["lineset"] = ls

        # 사람 스켈레톤 선(LineSet)과 관절 구체(sphere) 저장소
        joint_storage = {}
        # 팔꿈치에서 손목 방향으로 그릴 arrow(반직선) 저장소
        arrow_storage = {"arrow": None, "prev_transform": np.eye(4)}

        time.sleep(2)

        while True:
            capture = device.update()
            ret, frame = capture.get_color_image()
            if not ret:
                print("Kinect 프레임을 가져올 수 없습니다.")
                continue

            try:
                body_frame = body_tracker.update()
                body_pos = get_kinect_body_positions(body_frame)
                if body_pos is not None:
                    # 스켈레톤 선과 각 관절 구체 업데이트
                    update_body_positions(vis, body_pos, joint_storage)
                    # 팔꿈치에서 손목 방향의 arrow(반직선) 그리기
                    draw_pointing_arrow(vis, body_pos, arrow_storage)
                    # arrow 길이를 1500으로 연장하여 충돌 계산
                    start, end, arrow_radius = get_arrow_segment_from_body(body_pos)
                    if start is not None and end is not None:
                        # 첫 번째 충돌하는 바운딩박스 정보를 반환 (없으면 None)
                        first_collision = check_cylinder_hit_instances(
                            start, end, arrow_radius, instance_boxes
                        )
                        if first_collision is not None:
                            # 첫 충돌 대상의 대표 색으로 arrow 색 변경
                            arrow_storage["arrow"].paint_uniform_color(
                                first_collision["color"]
                            )
                            print(
                                "첫 충돌 대상 :",
                                first_collision["kor_class_name"],
                                " / ID :",
                                first_collision["instance_label"],
                            )
                        else:
                            # 충돌 없으면 노란색 유지
                            arrow_storage["arrow"].paint_uniform_color([1, 1, 0])
                else:
                    draw_pointing_arrow(vis, None, arrow_storage)
            except Exception as e:
                print("Skeleton 처리 오류:", e)
                continue

            vis.poll_events()
            vis.update_renderer()

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        device.close()
        cv2.destroyAllWindows()
        vis.destroy_window()
        print("정상적으로 종료되었습니다.")

    except Exception as e:
        print("예외 발생:", e)
    finally:
        cv2.destroyAllWindows()
        try:
            device.close()
        except:
            pass


if __name__ == "__main__":
    main()
