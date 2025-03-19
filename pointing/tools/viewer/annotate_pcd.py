import open3d as o3d
import json
import numpy as np
import sys


def annotate_pointcloud(pcd_file, json_file):
    # PCD 파일 읽기
    pcd = o3d.io.read_point_cloud(pcd_file)

    # 포인트 클라우드에 컬러 정보가 없는 경우 기본값(흰색)을 할당
    if not pcd.has_colors():
        num_points = np.asarray(pcd.points).shape[0]
        default_colors = np.ones((num_points, 3))  # 모두 흰색
        pcd.colors = o3d.utility.Vector3dVector(default_colors)

    # JSON 파일 읽기
    with open(json_file, "r") as f:
        annot = json.load(f)

    # JSON 파일에는 "data"와 "colors" 필드가 있어야 함
    if "data" not in annot:
        print("JSON 파일에 'data' 필드가 없습니다.")
        return pcd, {}
    if "colors" not in annot:
        print("JSON 파일에 'colors' 필드가 없습니다.")
        return pcd, {}

    # "data" 필드는 각 주석의 정보 [index, b, g, r]를 담고 있음
    data = annot["data"]
    colors_np = np.asarray(pcd.colors)

    # 원래의 b, g, r 값을 저장할 딕셔너리 (키: 인덱스, 값: (b, g, r))
    annotation_map = {}

    for ann in data:
        if len(ann) != 4:
            continue
        idx, b, g, r = ann
        annotation_map[idx] = (b, g, r)  # JSON의 원래 값 (0~255)
        # 인덱스 범위 확인
        if idx < 0 or idx >= colors_np.shape[0]:
            print(f"인덱스 {idx}는 범위를 벗어났습니다.")
            continue
        # Open3D는 색상 순서가 [r, g, b]이며, 0~1 범위의 float 값을 사용하므로 변환 필요
        colors_np[idx] = [r / 255.0, g / 255.0, b / 255.0]

    # 수정된 색상 정보를 포인트 클라우드에 다시 적용
    pcd.colors = o3d.utility.Vector3dVector(colors_np)
    return pcd, annotation_map


def interactive_color_check(pcd, annotation_map):
    """
    VisualizerWithEditing을 사용하여 사용자가 원하는 점들을 선택하고,
    선택된 점 인덱스에 대해 JSON 파일의 원래 b, g, r 값을 터미널에 출력합니다.

    사용법:
    - 창이 열리면 원하는 점들을 왼쪽 마우스 클릭으로 선택합니다.
    - 선택을 마친 후 'Q' 또는 'Esc' 키를 눌러 창을 닫으면,
      선택된 점의 인덱스와 원래의 JSON 색상 값이 출력됩니다.
    """
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name="Point Picker (선택 후 'Q' 눌러 종료)", width=800, height=600
    )
    vis.add_geometry(pcd)
    vis.run()  # 사용자의 입력을 기다림
    # shift + left click
    picked_indices = vis.get_picked_points()
    vis.destroy_window()

    if not picked_indices:
        print("선택된 점이 없습니다.")
    else:
        print("선택된 점 인덱스:", picked_indices)
        for idx in picked_indices:
            if idx in annotation_map:
                b, g, r = annotation_map[idx]
                print(f"인덱스 {idx}의 JSON 색상 (B, G, R): ({b}, {g}, {r})")
            else:
                print(f"인덱스 {idx}는 JSON 데이터에 없습니다.")


if __name__ == "__main__":
    pcd_file = r"C:\Users\sumin\Documents\GitHub\JARVIS\pointing\data\experiment.pcd"
    json_file = r"C:\Users\sumin\Documents\GitHub\JARVIS\pointing\data\experiment.json"

    annotated_pcd, annotation_map = annotate_pointcloud(pcd_file, json_file)

    # 결과를 시각화 (주석이 적용된 포인트 클라우드)
    o3d.visualization.draw_geometries(
        [annotated_pcd], window_name="Annotated PointCloud"
    )

    # 인터랙티브하게 점 선택 후, JSON의 b, g, r 값 확인
    interactive_color_check(annotated_pcd, annotation_map)
