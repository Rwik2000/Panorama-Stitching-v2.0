import cv2
import numpy as np

# load image with alpha channel
img = cv2.imread('object.png', cv2.IMREAD_UNCHANGED)

# extract alpha channel
alpha = img[:, :, 3]

# threshold alpha channel
alpha = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]
print(alpha)
# print(np.where(alpha==255))
# save output
cv2.imwrite('object_alpha.png', alpha)

# Display various images to see the steps
cv2.imshow('alpha',alpha)
cv2.waitKey(0)
cv2.destroyAllWindows()