import numpy as np
import cv2
from matplotlib import pyplot as plt

#初始化操作
img = cv2.imread('./images/red1.png')
rows,cols,channels = img.shape
img = img[rows//8:,cols//5:cols//5*4]
cv2.imshow("Origin",img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#去除白色的噪声
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
cv2.imshow("1",opening)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
cv2.imshow("2",sure_bg)

img[sure_bg == 0] = [255,255,255]
cv2.imshow("output",img)

# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# cv2.imshow("3",sure_fg)
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)
# cv2.imshow("unknown",unknown)

# # Marker labelling
# ret, markers = cv2.connectedComponents(sure_fg)
# # Add one to all labels so that sure background is not 0, but 1
# markers = markers+1
# # Now, mark the region of unknown with zero
# markers[unknown==255] = 0

# markers = cv2.watershed(img,markers)
# img[markers == 0] = [255,0,0]

# cv2.imshow("output",img)

cv2.waitKey(0)