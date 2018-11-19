import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('./images/red1.png')
rows,cols,channels = img.shape
img = img[60:,cols//4:cols//4*3]
cv2.imshow("Origin",img)

#简单阈值法
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# blur= cv2.GaussianBlur(gray,(5,5),0)
# #THRESH_BINARY_INV是反向二值阈值化，THRESH_OTSU是OTSU二值化
# ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# threshResult = img.copy()
# threshResult[thresh == 0] = [255,255,255]
# cv2.imshow("Threshold",threshResult)

#色度阈值法
I = img[:,:,2] - (img[:,:,0] + img[:,:,1]) // 2
cv2.imshow("I",I)
hist_I = cv2.calcHist([I],[0],None,[256],[0,256])
plt.plot(hist_I)
plt.show()

ravel = I.ravel()
myMax = int(max(ravel))
myMin = int(min(ravel))
T = (myMax + myMin) // 2
temp = T

while True:
    f = np.ma.masked_greater(ravel,temp)
    leftAvg = np.mean(f)
    g = np.ma.masked_less(ravel,temp)
    rightAvg = np.mean(g)
    T = (leftAvg + rightAvg) // 2
    if temp == T:
        break
    temp = T

print(T)
colorResult = img.copy()
colorResult[I >= T] = [255,255,255]
cv2.imshow("Color",colorResult)

cv2.waitKey(0)

