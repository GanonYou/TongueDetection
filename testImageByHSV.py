import numpy as np
import cv2
import sys
#from matplotlib import pyplot as plt

# 函数功能: 人脸检测
def faceDetection(image_init):
    #获取图片的原始宽度和高度
    image_init_height, image_init_width = image_init.shape[:2]
    #如果原始图片的宽度和高度都大于500，则对原始图像进行缩小到（500,500 * image_init_height / image_init_width)
    # if image_init_height > 500 and image_init_width > 500:
    #     image = cv2.resize(
    #         image_init, (500, 500 * image_init_height // image_init_width),
    #         interpolation=cv2.INTER_CUBIC)
    # else:
    #     image = image_init

    #原始图片转为灰度图再进行人脸检测
    image = image_init
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #加载分类器
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=6, minSize=(3, 3), flags=4)

    #print("发现{0}个人脸!".format(len(faces)))
    count = 0
    face_x = 0
    face_y = 0
    face_width = 0
    face_height = 0
    for (x, y, w, h) in faces:
        count += 1
        #保存人脸定位框的宽度和高度以及定位框的左上角像素点，如果识别出来的人脸有多个，则重拍
        face_x = x
        face_y = y
        face_width = w
        face_height = h
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

    if count != 1:
        return False,0,0,0,0

    if count == 1:
        #对原定位框进行拉伸
        image_height, image_width = image.shape[:2]
        # if(face_x-0.01*face_width)>0:
        #     face_x = face_x-0.01*face_width
        # else:
        #     face_x = 0
        if(face_y-0.1*face_height)>0:
            face_y = face_y-0.1*face_height
        else:
            face_y = 0

        # if (face_x + 1.01 * face_width) > image_width:
        #     face_width = image_width - face_x
        # else:
        #     face_width = 1.01 * face_width
        if (face_y + 1.4 * face_height) > image_height:
            face_height = image_height - face_y
        else:
            face_height = 1.4 * face_height
        # cv2.rectangle(image, (int(face_x), int(face_y)), (int(face_x + face_width), int(face_y + face_height)), (0, 255, 0), 1)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return True,int(face_x),int(face_y),int(face_width),int(face_height)

# 函数功能:求出输入矩阵中所有不为0元素的最大值与最小值
# 输入:矩阵 
# 输出:最大值与最小值
def getMyMaxAndMin(I):
    myMax = 0
    myMin = 255
    rows,columns= I.shape
    for i in range(rows):
        for j in range(columns):
            if I[i][j] != 0:
                if I[i][j] > myMax:
                    myMax = I[i][j]
                if I[i][j] < myMin:
                    myMin = I[i][j]
    return myMax,myMin

# 函数功能:迭代法确定色度法中的阈值，忽略矩阵中所有为0的元素
# 输入:矩阵 
# 输出:符合要求的阈值
def iterativeThreshold(I):
    rows,columns= I.shape
    myMax,myMin = getMyMaxAndMin(I)
    T = (myMax + myMin) / 2
    temp = T

    while True:
        #计算所有小于temp且不为0元素的平均值
        leftCount = 0
        leftSum = 0
        for i in range(rows):
            for j in range(columns):
                if I[i][j] != 0 and I[i][j] < temp:
                    leftCount += 1
                    leftSum += I[i][j]
        if leftCount != 0:
            leftAvg = leftSum / leftCount
        else:
            leftAvg = 0

        #计算所有大于temp且不为0元素的平均值
        rightCount = 0
        rightSum = 0
        for i in range(rows):
            for j in range(columns):
                if I[i][j] != 0 and I[i][j] >= temp:
                    rightCount += 1
                    rightSum += I[i][j]
        if rightCount != 0:
            rightAvg = rightSum / rightCount
        else:
            rightAvg = 0

        T = (leftAvg + rightAvg) / 2
        if temp == T:
            break
        temp = T

    return T

# 函数功能:将输入图像拆分为H,S,V通道,并进行一系列处理提出舌头的轮廓
# 输入:原始图像 
# 输出:二值化的closing矩阵，舌头区域为255，非舌头区域为0
def hsvDeal(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    cv2.imshow('h',h)
    cv2.imshow('v',v)
    #二值化H通道与V通道
    _,thresh_h = cv2.threshold(h,127,255,cv2.THRESH_BINARY)
    _,thresh_v = cv2.threshold(v,127,255,cv2.THRESH_BINARY)
    #对H通道与V通道进行“与”运算
    hAndV = cv2.bitwise_and(thresh_h,thresh_v)
    #进行形态学“闭”运算，先膨胀后腐蚀
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(hAndV, cv2.MORPH_CLOSE, kernel)
    #返回提取出的舌头轮廓
    return closing

# 函数功能:色度阈值法判断图像是否是舌头
# 输入:原图像
# 输出:bool值
def tongueColorDetect(img):
    #提取出舌头区域
    closing = hsvDeal(img)
    #舌头的掩膜,舌头区域为1，非舌头区域为0
    mask = closing.copy() / 255
    #计算“舌头”区域的占整个图像的比例，太低的话说明不是舌头
    effectiveRate = np.sum(mask == 1) / (mask.shape[0] * mask.shape[1])
    #非舌头区域置为白色
    tongueArea = img.copy()
    tongueArea[mask == 0] = [255,255,255]
    cv2.imshow('Tongue',tongueArea)

    #色度阈值法，I = R - (G + B) / 2
    #分离舌质与舌苔
    I = tongueArea[:,:,2] - (tongueArea[:,:,0] + tongueArea[:,:,1]) / 2
    I = I * mask

    iterT = iterativeThreshold(I)   #调用函数获得阈值
    colorResult = tongueArea.copy()
    colorResult[I >= iterT] = [255,255,255]
    cv2.imshow("Split",colorResult)
    avg_r = np.mean(tongueArea[(I > 0) & (I < iterT),2])
    avg_g = np.mean(tongueArea[I >= iterT,1])
    avg_compare = np.mean(I[(I > 0) & (I < iterT)])
    print("The average value of compare is:" + str(avg_compare))
    print("The average value of r is:" + str(avg_r))
    print("The average value of g is:" + str(avg_g))
    print("The rate of the effective area is:" + str(effectiveRate))
   
    #可以调节的四个参数
    #舌质部位的r平均值;r分量突出程度的平均值;舌质部位的g分量平均值;舌头区域的有效占比effectiveRate
    if avg_r >= 130 and avg_compare >= 25 and avg_g < 200 and effectiveRate >= 0.05:
        return True
    else:
        return False


# -------- 主程序开始 --------
# 从人脸中提取的
frame = cv2.imread('./Images/me2-corrected.JPG')
isFace,face_x,face_y,face_width,face_height = faceDetection(frame)
if isFace == 0:
    print('未监测到人脸')
else:
    frame = frame[face_y+face_height*2//3 : face_y+face_height*9//10,face_x+face_width*1//3:face_x+face_width*2//3]
    cv2.imshow('Origin',frame)
    isTongue = tongueColorDetect(frame)
    if isTongue == True:
        cv2.putText(frame, 'It is a tongue!',(5,30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2, 8,0)    
    else:
        cv2.putText(frame, 'It is not a tongue!',(5,30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2, 8,0)    

    cv2.imshow('Frame',frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# # 直接提取的
# frame = cv2.imread('./Images/good5.png')
# cv2.imshow('Origin',frame)
# isTongue = tongueColorDetect(frame)
# if isTongue == True:
#     cv2.putText(frame, 'It is a tongue!',(5,30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2, 8,0)    
# else:
#     cv2.putText(frame, 'It is not a tongue!',(5,30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2, 8,0)    

# cv2.imshow('Frame',frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()