import cv2

# 待检测的图片路径
imagepath = "Images/me2.jpg"

image_init = cv2.imread(imagepath)

#获取图片的原始宽度和高度
image_init_height, image_init_width = image_init.shape[:2]

#如果原始图片的宽度和高度都大于500，则对原始图像进行缩小到（500,500 * image_init_height / image_init_width)
if image_init_height > 500 and image_init_width > 500:
    image = cv2.resize(
        image_init, (500, 500 * image_init_height // image_init_width),
        interpolation=cv2.INTER_CUBIC)
else:
    image = image_init

#原始图片转为灰度图再进行人脸检测
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

faces = face_cascade.detectMultiScale(
    gray, scaleFactor=1.15, minNeighbors=6, minSize=(3, 3), flags=4)

print("发现{0}个人脸!".format(len(faces)))
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

if count == 0:
    print("拍摄的照片检测不出人脸，请重拍!")
if count > 1:
    print("拍摄的照片中存在多个人脸，请重拍!")

if count == 1:
    print("拍摄成功!")
    #对原定位框进行拉伸
    image_height, image_width = image.shape[:2]
    if(face_x-0.05*face_width)>0:
          face_x = face_x-0.05*face_width
    else:
          face_x = 0
    if(face_y-0.1*face_height)>0:
          face_y = face_y-0.1*face_height
    else:
          face_y = 0

    if (face_x + 1.05 * face_width) > image_width:
        face_width = image_width - face_x
    else:
        face_width = 1.1 * face_width
    if (face_y + 1.4 * face_height) > image_height:
        face_height = image_height - face_y
    else:
        face_height = 1.4 * face_height
    cv2.rectangle(image, (int(face_x), int(face_y)), (int(face_x + face_width), int(face_y + face_height)), (0, 255, 0), 1)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()