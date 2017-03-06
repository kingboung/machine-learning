#usr/bin/env python2
#-*- coding:utf-8 -*-

import cv2

window=cv2.namedWindow('Face detect(Press \'q\' to quit)')

#打开一号摄像头
video=cv2.VideoCapture(0)

#读取一帧图像
sucess,frame=video.read()

"""创建haar级联，定义分类器"""
facecascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while sucess:
    sucess, frame = video.read()

    #转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 识别人脸（使用级联分类器）
    """
    faces表示检测到的人脸目标序列
    1.05表示每次图像尺寸减小的比例为1.05
    5表示每一个目标至少要被检测到5次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸)
    """
    faces = facecascade.detectMultiScale(gray, 1.05, 5)

    # 把识别到的人脸框出来
    for (x, y, w, h) in faces:
        image = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    #显示出来
    cv2.imshow('Face detect(Press \'q\' to quit)',frame)
    key=cv2.waitKey(1)

    #监听到按键‘q’退出
    if key==ord('q'):
        break

#释放资源
video.release()
cv2.destroyAllWindows()