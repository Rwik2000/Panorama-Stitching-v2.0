import numpy as np
import cv2
from Warp import Warp
invH = np.array([[ 8.57468067e-01, -3.14290344e-02, -7.14854444e+02],
        [ 3.34238196e-01,  9.45894515e-01, -4.24502432e+02],
        [ 3.12113910e-03,  8.64500623e-05, -1.63260753e+00]])
H = np.array([[-1.83497389e+00, -1.37674493e-01,  8.39260125e+02],
     [-9.48479968e-01,  1.01177271e+00,  1.52225897e+02],
     [-3.55823724e-03, -2.09623822e-04,  1.00000000e+00]])

image = cv2.imread("Dataset/I2/2_5.JPG")
dsc = Warp()
dsc.xOffset = 4000 ; dsc.yOffset = 4000
j = dsc.InvWarpPerspective(image, invH, H,  (5000, 5000))
l = cv2.warpPerspective(image, H, (5000, 5000))
cv2.imwrite("tp.jpg", np.uint8(j))
cv2.imwrite("tp1.jpg", np.uint8(l))