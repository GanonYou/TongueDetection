import numpy as np
import cv2
from matplotlib import pyplot as plt

#函数功能:迭代法确定色度法中的阈值
#输入:array数组，输出:符合要求的阈值
def iterativeThreshold(I):
    ravel = I.ravel()
    myMax = float(max(ravel))
    myMin = float(min(ravel))
    T = (myMax + myMin) / 2
    temp = T

    while True:
        f = np.ma.masked_greater(ravel,temp)
        leftAvg = np.mean(f)   #求ravel中所有小于temp值的平均值
        g = np.ma.masked_less(ravel,temp)
        rightAvg = np.mean(g)   #求ravel中所有大于temp值的平均值
        T = (leftAvg + rightAvg) / 2
        if temp == T:
            break
        temp = T
    return T

#初始化操作
img = cv2.imread('./images/real3.png')
rows,cols,channels = img.shape
img = img[50:,cols//4:cols//4*3]
cv2.imshow("Origin",img)

#OTSU阈值法
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur= cv2.GaussianBlur(gray,(5,5),0)
#THRESH_BINARY_INV是反向二值阈值化，THRESH_OTSU是OTSU二值化
ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
threshResult = img.copy()
threshResult[thresh == 0] = [255,255,255]
cv2.imshow("OTSU",threshResult)

#色度阈值法，I = R - (G + B) / 2
I = img[:,:,2] - (img[:,:,0] + img[:,:,1]) / 2
#绘制I的直方图
# hist_I = cv2.calcHist([I],[0],None,[256],[0,256])
# plt.plot(hist_I)
# plt.show()
iterT = iterativeThreshold(I)   #调用函数获得阈值
colorResult = img.copy()
colorResult[I >= iterT] = [255,255,255]
cv2.imshow("Color",colorResult)
r = img[I < iterT,2]
g = img[I < iterT,1]

print("The average value of r is:" + str(np.mean(r) / 255))
print("The average value of g is:" + str(np.mean(g) / 255))

cv2.waitKey(0)

