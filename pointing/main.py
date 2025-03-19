import cv2
import time
import numpy as np
import open3d as o3d

from modules.parameters import *
from modules.aruco_calibration import get_transformation_matrix
from modules.kinect_manager import initialize_kinect, reset_kinect_devices
from modules.visualization import setup_scene
from modules.body_tracking import (
    get_kinect_body_positions,
    update_body_positions,
    draw_pointing_arrow,
    get_arrow_segment_from_body,
)
from modules.bounding_box_collision import (
    check_cylinder_hit_all_instances,
    create_aabb_lineset,
)
from modules.process import process_frame

# 전역 변수: 최신 프레임에서 충돌된 인스턴스 ID 리스트를 저장합니다.
collision_list_global = None
should_quit = False


def save_collision_callback(vis):
    """
    'S' 키가 눌리면 현재 충돌된 인스턴스 리스트를 텍스트 파일에 저장합니다.
    """
    global collision_list_global
    if collision_list_global is not None:
        with open("result/collision_log.txt", "a", encoding="utf-8") as f:
            f.write(str(collision_list_global) + "\n")
        print("충돌 리스트가 collision_log.txt에 저장되었습니다.")
    else:
        print("저장할 충돌 리스트가 없습니다.")
    return False  # 기본 처리를 계속 진행


def quit_callback(vis):
    """Q
    'Q' 키가 눌리면 Visualizer를 종료합니다.
    """
    global should_quit
    should_quit = True
    return False


def main():
    global collision_list_global
    try:
        # Kinect 초기화
        try:
            reset_kinect_devices()
            device, body_tracker = initialize_kinect()
        except Exception as e:
            print("Kinect 초기화 오류:", e)
            return

        # setup_scene을 이용해 pcd, instance_boxes, labels 불러오기
        # (이때 vis는 VisualizerWithKeyCallback 객체여야 함)
        try:
            vis, pcd, instance_boxes, labels = setup_scene(
                point_cloud_file="data\\experiment.pcd",
                label_file="data\\experiment.json",
            )
        except Exception as e:
            print("포인트 클라우드 로드 오류:", e)
            return

        # Open3D 키 이벤트 콜백 등록 ('S'와 'Q')
        vis.register_key_callback(ord("S"), save_collision_callback)
        vis.register_key_callback(ord("Q"), quit_callback)
        vis.register_key_callback(ord("s"), save_collision_callback)
        vis.register_key_callback(ord("q"), quit_callback)

        # 각 인스턴스의 AABB 라인셋 생성 및 vis에 추가
        for inst in instance_boxes:
            ls = create_aabb_lineset(inst)
            vis.add_geometry(ls)
            inst["lineset"] = ls

        # 스켈레톤(라인셋, 구체)와 arrow 업데이트를 위한 저장소 초기화
        joint_storage = {}
        arrow_storage = {"arrow": None, "prev_transform": np.eye(4)}

        # pcd의 원본 색상 백업 (N x 3 numpy array)
        original_colors = np.asarray(pcd.colors).copy()
        # 이전 프레임에 충돌된 인스턴스 ID (없으면 None)
        prev_instance_ids = None

        time.sleep(2)

        # 메인 루프
        while True:
            frame, overlay, prev_instance_ids, prev_instance_name = process_frame(
                device,
                body_tracker,
                vis,
                pcd,
                instance_boxes,
                labels,
                joint_storage,
                arrow_storage,
                original_colors,
                prev_instance_ids,
            )
            if frame is None:
                continue

            # 최신 충돌 인스턴스 리스트를 전역 변수에 저장
            collision_list_global = prev_instance_name

            # Open3D 이벤트 처리 (키 콜백 포함)
            vis.poll_events()
            vis.update_renderer()

            # OpenCV 창에 overlay 표시
            cv2.imshow("Overlay", overlay)
            # key = cv2.waitKey(30) & 0xFF
            # if key == ord("q"):
            #     vis.close()
            #     break

            # Open3D 창이 닫히면 루프 종료
            if should_quit:
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
        except Exception:
            pass


if __name__ == "__main__":
    main()
