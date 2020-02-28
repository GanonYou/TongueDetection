import numpy as np
import cv2
import os, sys

# ----- 预定义参数 ------
# 输入图像像素标准
IMG_WIDTH = 1080
IMG_HEIGHT = 1440
# 舌上表面参数
UP_AVG_R = 107
UP_AVG_COMPARE = 25
UP_EFFECTIVE_RATE = 0.5
# 舌下表面参数
DOWN_AVG_R = 80
DOWN_AVG_COMPARE = 20
DOWN_EFFECTIVE_RATE = 0.5
# ---------------------


#函数功能:迭代法确定色度法中的阈值
#输入:array数组，输出:符合要求的阈值
def iterativeThreshold(I):
    ravel = I.ravel()
    myMax = float(max(ravel))
    myMin = float(min(ravel))
    T = (myMax + myMin) / 2
    temp = T
    while True:
        f = np.ma.masked_greater(ravel, temp)
        leftAvg = np.mean(f)  #求ravel中所有小于temp值的平均值
        g = np.ma.masked_less(ravel, temp)
        rightAvg = np.mean(g)  #求ravel中所有大于temp值的平均值
        T = (leftAvg + rightAvg) / 2
        if temp == T:
            break
        temp = T
    return T


#函数功能: 判断是否是舌头上侧(即伸舌头的情况)
def isTongue(img):
    #色度阈值法，I = R - (G + B) / 2
    I = img[:, :, 2] - (img[:, :, 0] + img[:, :, 1]) / 2
    iterT = iterativeThreshold(I)  #调用函数获得阈值
    tongueArea = img.copy()
    effectiveRate = np.sum(I < iterT) / (I.shape[0] * I.shape[1])
    #cv2.imshow("Color",tongueArea)
    avg_r = np.mean(tongueArea[(I > 0) & (I < iterT), 2])
    avg_g = np.mean(tongueArea[I >= iterT, 1])
    avg_b = np.mean(tongueArea[I >= iterT, 0])
    avg_compare = np.mean(I[(I > 0) & (I < iterT)])
    print("The average value of compare is:" + str(avg_compare))
    print("The average value of r is:" + str(avg_r))
    print("The average value of g is:" + str(avg_g))
    print("The average value of b is:" + str(avg_b))
    print("The rate of the effective area is:" + str(effectiveRate))

    #可以调节的四个参数
    #舌质部位的r平均值;r分量突出程度的平均值;舌头区域的有效占比effectiveRate
    if avg_r >= UP_AVG_R and avg_compare >= UP_AVG_COMPARE and effectiveRate >= UP_EFFECTIVE_RATE:
        return True
    else:
        return False


# 函数功能: 判断是否是舌头下侧(即翘舌头的情况)
def isTongueDown(img):
    #色度阈值法，I = R - (G + B) / 2
    I = img[:, :, 2] - (img[:, :, 0] + img[:, :, 1]) / 2
    iterT = iterativeThreshold(I)  #调用函数获得阈值
    tongueArea = img.copy()
    effectiveRate = np.sum(I < iterT) / (I.shape[0] * I.shape[1])
    #cv2.imshow("Color",tongueArea)
    avg_r = np.mean(tongueArea[(I > 0) & (I < iterT), 2])
    avg_g = np.mean(tongueArea[I >= iterT, 1])
    avg_b = np.mean(tongueArea[I >= iterT, 0])
    avg_compare = np.mean(I[(I > 0) & (I < iterT)])
    print("The average value of compare is:" + str(avg_compare))
    print("The average value of r is:" + str(avg_r))
    print("The average value of g is:" + str(avg_g))
    print("The average value of b is:" + str(avg_b))
    print("The rate of the effective area is:" + str(effectiveRate))

    #可以调节的四个参数
    #舌质部位的r平均值;r分量突出程度的平均值;舌头区域的有效占比effectiveRate
    if avg_r >= DOWN_AVG_R and avg_compare >= DOWN_AVG_COMPARE and effectiveRate >= DOWN_EFFECTIVE_RATE:
        return True
    else:
        return False


# 函数功能:嘴唇检测
# 检测到嘴唇返回True,否则为False
def mouthDetection(img):
    mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
    if mouth_cascade.empty():
        raise IOError('Unable to load the mouth cascade classifier xml file')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
    count = 0
    for (x, y, w, h) in mouth_rects:
        count += 1
        y = int(y - 0.15 * h)
        return True, x, y, w, h

    return False, 0, 0, 0, 0


# 函数功能: 人脸检测
# 若没检测到人脸，则返回 False,0,0,0,0
# 若检测到人脸，则返回 True,人脸位置的左上角x,y,以及人脸框的width,height
def faceDetection(image_init):
    #获取图片的原始宽度和高度
    #image_init_height, image_init_width = image_init.shape[:2]
    # if image_init_height > 500 and image_init_width > 500:
    #     image = cv2.resize(
    #     image_init, (500, 500 * image_init_height // image_init_width),
    #     interpolation=cv2.INTER_CUBIC)
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

    if count > 1:
        print('检测到多个人脸!')
        return False, 0, 0, 0, 0, count
    elif count == 0:
        print('未检测到人脸!')
        return False, 0, 0, 0, 0, count
    else:
        #对原定位框进行拉伸
        image_height, image_width = image.shape[:2]
        # if(face_x-0.01*face_width)>0:
        #     face_x = face_x-0.01*face_width
        # else:
        #     face_x = 0
        if (face_y - 0.1 * face_height) > 0:
            face_y = face_y - 0.1 * face_height
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
        # 若检测到人脸，则返回 True,人脸位置的左上角x,y,以及人脸框的width,height
        return True, int(face_x), int(face_y), int(face_width), int(
            face_height), count


# 函数功能: 用于检测输入图像是否为面部/伸舌头/翘舌头
# 输入参数: img - 输入图像
#          type - 0表示检测面部;1表示检测是否为伸舌头;2表示是否为舌下
# 返回参数:  (0,0,0,0,0)：输入图像检测成功
#           (1,0,0,0,0)：输入图像像素不达标
#           (2,0,0,0,0)：输入的图像不是伸舌头/翘舌头
#           (3,0,0,0,0)：输入参数type不符合要求
#           (4,0,0,0,0)：输入图像未检测到人脸
#           (5,0,0,0,0)：输入图像存在多个人脸
#           (0,x,y,w,h)：当type为0且检测到人脸时会返回人脸的位置信息，分别代表人脸位置左上角x,y，以及人脸框的width和height
def inputImageCheck(img, type=0):
    # 判断输入参数type是否符合要求
    if type != 0 and type != 1 and type != 2:
        print('输入参数不合法!')
        return 3, 0, 0, 0, 0

    # 判断输入图像像素是否达标
    if img.shape[1] < IMG_WIDTH or img.shape[0] < IMG_HEIGHT:
        print('输入像素不合法!')
        return 1, 0, 0, 0, 0

    # 判断是否是面部
    isFace, face_x, face_y, face_width, face_height, count = faceDetection(img)
    if type == 0 and isFace == 1:
        print('检测到人脸!')
        return 0, face_x, face_y, face_width, face_height
    if isFace == 0:
        #print('未检测到人脸或者检测出多个人脸!')
        if count == 0:
            return 4, 0, 0, 0, 0
        if count > 1:
            return 5, 0, 0, 0, 0

    # 判断是否是舌部
    tongueImg = img[face_y + face_height * 2 // 3:face_y +
                    face_height * 9 // 10, face_x +
                    face_width * 1 // 3:face_x + face_width * 2 // 3]
    faceImg = img[face_y:face_y + face_height, face_x:face_x + face_width]
    mouthArea = img[face_y + face_height // 2:face_y + face_height, face_x:face_x + face_width]
    cv2.imshow('roi',mouthArea)
    isMouth, mouth_x, mouth_y, mouth_width, mouth_height = mouthDetection(mouthArea)
    #mouthImg = faceImg[mouth_y:mouth_y + mouth_height, mouth_x:mouth_x + mouth_width]
    cv2.waitKey(0)
    if type == 1:
        if isMouth == False and isTongue(tongueImg):
            print('检测到舌头!')
            return 0, 0, 0, 0, 0
        else:
            print('未检测到舌头!')
            return 2, 0, 0, 0, 0
    if type == 2:
        if isTongueDown(tongueImg):
            print('检测到舌头!')
            return 0, 0, 0, 0, 0
        else:
            print('未检测到舌头!')
            return 2, 0, 0, 0, 0


if __name__ == '__main__':
    imgPath = sys.argv[1]
    inputType = int(sys.argv[2])
    if os.path.exists(imgPath) == False:
        print('图片路径错误!')
        sys.exit(1)
    testImg = cv2.imread(imgPath)
    print(inputImageCheck(testImg, inputType))
    '''
    imgPath = 'Images/me10.jpg'
    testImg = cv2.imread(imgPath)
    if testImg is None:
        print("图片路径错误")
    else:
        inputType = 1
        _ = inputImageCheck(testImg, inputType)
        print(_)
    '''