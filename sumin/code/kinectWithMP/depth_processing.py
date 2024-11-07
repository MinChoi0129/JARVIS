def process_depth_information(hand_landmarks, depth_image, image_width, image_height):
    hand_data = []
    depth_height, depth_width = depth_image.shape[:2]

    for idx, landmark in enumerate(hand_landmarks.landmark):
        # 2D 랜드마크를 깊이 이미지의 픽셀 좌표로 변환
        dx, dy = int(landmark.x * depth_height), int(landmark.y * depth_width)

        x, y = int(landmark.x * image_height), int(landmark.y * image_width)

        # 깊이 이미지에서 해당 좌표의 깊이 값을 가져옴
        try:
            depth = depth_image[dy, dx]
        except:
            pass

        # 3D 위치 데이터 생성
        hand_data.append({'index': idx, 'x': x, 'y': y, 'depth': depth})

    return hand_data
