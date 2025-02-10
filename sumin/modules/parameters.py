import cv2
import numpy as np

fx = 608.427060
fy = 608.427060
cx = 640.000000
cy = 360.000000
k1 = 0.077901
k2 = -0.053721
p1 = -0.001677
p2 = -0.001842

marker_length = 168

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()

# 카메라 행렬과 왜곡 계수
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist_coeffs = np.array([k1, k2, p1, p2, 0])
