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

#函数功能: 判断是否是舌头
def isTongue(img):
    #色度阈值法，I = R - (G + B) / 2
    I = img[:,:,2] - (img[:,:,0] + img[:,:,1]) / 2
    iterT = iterativeThreshold(I)   #调用函数获得阈值
    tongueArea = img.copy()
    tongueArea[I >= iterT] = [255,255,255]
    effectiveRate = np.sum(I < iterT) / (I.shape[0] * I.shape[1])
    cv2.imshow("Color",tongueArea)
    avg_r = np.mean(tongueArea[(I > 0) & (I < iterT),2])
    avg_g = np.mean(tongueArea[I >= iterT,1])
    avg_compare = np.mean(I[(I > 0) & (I < iterT)])
    print("The average value of compare is:" + str(avg_compare))
    print("The average value of r is:" + str(avg_r))
    print("The average value of g is:" + str(avg_g))
    print("The rate of the effective area is:" + str(effectiveRate))

    #可以调节的四个参数
    #舌质部位的r平均值;r分量突出程度的平均值;舌头区域的有效占比effectiveRate
    if avg_r >= 90 and avg_compare >= 20 and effectiveRate >= 0.8:
        return True
    else:
        return False

if __name__ == '__main__':
    #初始化操作
    img = cv2.imread('./images/face.png')
    rows,cols,channels = img.shape
    img = img[50:,cols//5:cols//5*4]
    cv2.imshow("Origin",img)
    if isTongue(img) == True:
        print("It is a tongue")
    else:
        print("It is not a tongue")