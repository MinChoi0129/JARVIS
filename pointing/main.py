### main.py

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


def main():
    try:
        # Kinect 초기화
        try:
            reset_kinect_devices()
            device, body_tracker = initialize_kinect()
        except Exception as e:
            print("Kinect 초기화 오류:", e)
            return

        # # Transformation Matrix 계산
        # try:
        #     C2W = get_transformation_matrix(
        #         device,
        #         aruco_dict,
        #         aruco_params,
        #         marker_length,
        #         camera_matrix,
        #         dist_coeffs,
        #     )
        #     print(C2W)
        # except Exception as e:
        #     print(e, f">>>>>>>>>>> Transformation Matrix 계산 오류")
        #     return

        # 포인트 클라우드 및 시각화 장면 설정 (pcd, instance_boxes, labels 포함)
        try:
            vis, pcd, instance_boxes, labels = setup_scene(
                point_cloud_file="data/702_coord.npy",
                label_file="data/702_instance_with_class_pred.npy",
            )
        except Exception as e:
            print("포인트 클라우드 로드 오류:", e)
            return

        # 각 인스턴스의 AABB 라인셋 생성 및 추가
        for inst in instance_boxes:
            ls = create_aabb_lineset(inst)
            vis.add_geometry(ls)
            inst["lineset"] = ls

        # 스켈레톤(라인셋, 구체)과 arrow 업데이트를 위한 저장소 초기화
        joint_storage = {}
        arrow_storage = {"arrow": None, "prev_transform": np.eye(4)}

        # pcd의 원본 색상 백업 (N x 3 numpy array)
        original_colors = np.asarray(pcd.colors).copy()
        # 이전 프레임에서 충돌된 인스턴스 ID (없으면 None)
        prev_instance_id = None

        time.sleep(2)

        while True:
            frame, overlay, prev_instance_id = process_frame(
                device,
                body_tracker,
                vis,
                pcd,
                instance_boxes,
                labels,
                joint_storage,
                arrow_storage,
                original_colors,
                prev_instance_id,
            )
            if frame is None:
                continue

            vis.poll_events()
            vis.update_renderer()
            # cv2.imshow("Frame", frame)
            cv2.imshow("Overlay", overlay)
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
        except Exception:
            pass


if __name__ == "__main__":
    main()
