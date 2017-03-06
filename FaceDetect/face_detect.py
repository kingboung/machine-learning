#usr/bin/env python2
#-*- coding:utf-8 -*-

import cv2
import os
import time
import contextlib

timeList=[]

@contextlib.contextmanager
def timer():
    start=time.time()
    yield
    end=time.time()
    timeList.append(end-start)

def faceDetect():
    notMatch = 0
    sourceDir='FaceDatabase'
    targetDir='FaceDetectResult'

    global timeList
    total=len(os.listdir(sourceDir))

    """创建haar级联"""
    facecascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    """读取照片"""
    for img in os.listdir(sourceDir):
        image = cv2.imread(sourceDir+'/'+img)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        #识别人脸（使用级联分类器）
        """
        faces表示检测到的人脸目标序列
        1.05表示每次图像尺寸减小的比例为1.05
        5表示每一个目标至少要被检测到5次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸)
        """
        with timer():
            faces = facecascade.detectMultiScale(gray, 1.05, 5)

        #识别错误（这里单纯认为识别出一张脸便是识别成功）
        if len(faces)!=1:
            notMatch+=1.0
            cv2.imwrite(targetDir+'/' + img.split('.')[0] + '_notmatch.jpg', image)
            continue

        #把识别到的人脸框出来
        for (x,y,w,h) in faces:
            image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

        #将识别出人脸的照片保存下来
        cv2.imwrite(targetDir+'/'+img.split('.')[0] + '_match.jpg',image)

    with open('result.txt','w') as file:
        accuracy=(total-notMatch)/total
        timeMean=float(sum(timeList))/len(timeList)
        file.write('精度：'+str(accuracy)+'('+str(total-notMatch)+str(total)+')\n')
        file.write('平均用时：'+str(timeMean)+'\n')
        file.write('分别用时:'+'\n')
        for item in timeList:
            file.write(str(item)+'\n')

if __name__=='__main__':
    faceDetect()