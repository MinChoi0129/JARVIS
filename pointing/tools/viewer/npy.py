import numpy as np

# 불러올 .npy 파일 경로
file_path = "data/702_instance_with_class_pred.npy"

# npy 파일 로드
data = np.load(file_path)


# 데이터 내용 출력
print("데이터 내용:\n", data)

# 데이터의 형태(shape)와 자료형(dtype)도 함께 확인하기
print("데이터 shape:", data.shape)
print("데이터 dtype:", data.dtype)
