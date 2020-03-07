# TongueTest
## 依赖包
- opencv-python
- numpy
- sys,os

## 依赖文件

- haarcascade\_frontalface\_alt.xml
- haarcascade\_mcs\_mouth.xml

请将压缩包中的这两个文件和inputImageCheck.py放在同一个目录下。

## detectTongue.py
- Step 1: 通过HSV三通道的处理，提出舌体轮廓
- Step 2: 通过色度阈值法判断是否是舌头
- Step 3: 分离舌质舌苔

效果如下图所示:



## inputImageCheck.py
### 1.直接调用inputImageCheck(img,type) 函数
函数功能: 用于检测输入图像是否为面部/伸舌头/翘舌头

输入参数: 

- img：输入图像
- type：0表示检测面部; 1表示检测是否为伸舌头; 2表示是否为翘舌

返回参数: 

- (0,0,0,0,0)：输入图像检测成功
- (1,0,0,0,0)：输入图像像素不达标
- (2,0,0,0,0)：输入的图像不是伸舌头/翘舌头
- (3,0,0,0,0)：输入参数type不符合要求
- (4,0,0,0,0)：输入图像未检测到人脸
- (5,0,0,0,0)：输入图像存在多个人脸
- (0,x,y,w,h)：当type为0且检测到人脸时会返回人脸的位置信息，分别代表人脸位置左上角x,y，以及人脸框的width和height

### 2.命令行调用脚本
终端中调用脚本，type参数跟在路径之后:

```
python inputImageCheck.py 'PATH/TO/YOUR/IMAGE' 0
```
