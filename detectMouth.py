import cv2
import numpy as np

mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')

ds_factor = 0.5

WIDTH = 320
HEIGHT = 240
cap = cv2.VideoCapture(0)
cap.set(3,WIDTH)
cap.set(4,HEIGHT)

while True:
    ret, frame = cap.read()
    cv2.rectangle(frame, (WIDTH//3 - 5,HEIGHT//5*3 - 5), (WIDTH//3*2, HEIGHT), [0,255,0],1)
    newFrame = frame[HEIGHT//5*3:HEIGHT,WIDTH//3:WIDTH//3*2,:]
    gray = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY)
    
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
    count = 0
    for (x,y,w,h) in mouth_rects:
        count += 1
        y = int(y - 0.15*h)
        cv2.rectangle(newFrame, (x,y), (x+w,y+h), (0,255,0), 3)
        break

    cv2.imshow('Mouth Detector', frame)
    cv2.imshow('Output',newFrame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()