# -*- coding:utf-8 -*-

# regression

from numpy import *
import math

"""加载数据集并转换成能够程序能够处理的数据集"""
def loadDataSet(featFilename,labelFilename):
    dataResource=open(featFilename,'r')
    labelResource=open(labelFilename,'r')
    dataMat=[];labelMat=[]
    for line in dataResource.readlines():
        lineArr=[]
        numFeat=len(line.split('\t'))
        curLine=line.strip().split('\t')
        lineArr.append(1.0)
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
    for line in labelResource.readlines():
        curLine=line.strip().split()
        labelMat.append(float(curLine[0]))
    return dataMat,labelMat

"""划分数据集（for十折交叉验证）"""
def divideDataSet(dataMat,labelMat):
    # 以序号为索引，对应划分10个数据集
    dataSet=[]
    labelSet=[]
    for i in range(10):
        dataSet.append([])
        labelSet.append([])
    numberOfLine=len(dataMat)
    for i in range(numberOfLine):
        dataSet[i%10].append(dataMat[i])
        labelSet[i%10].append(labelMat[i])
    return dataSet,labelSet

"""按比例划分数据集"""
def divideRatioDataSet(dataMat,labelMat,percent):
    # 索引0对应的是训练集，索引1对应的是测试集
    dataSet=[]
    labelSet=[]
    dataSet.append([])
    dataSet.append([])
    labelSet.append([])
    labelSet.append([])

    if 1<percent<0:
        print 'The percent does not satisify demand.'
        return

    totalNumber=len(dataMat)
    number=int(totalNumber*percent)

    for i in range(number):
        dataSet[0].append(dataMat[i])
        labelSet[0].append(labelMat[i])
    for i in range(number,totalNumber):
        dataSet[1].append(dataMat[i])
        labelSet[1].append(labelMat[i])

    return dataSet,labelSet

"""线性回归"""
def standRegress(xArr,yArr):
    xMat=mat(xArr);yMat=mat(yArr).T
    xTx=xMat.T*xMat
    if linalg.det(xTx)==0.0:
        print "This matrix is singular,cannot do inverse"# 无法求矩阵的逆
        return
    ws=xTx.I*(xMat.T*yMat)
    return ws

"""逻辑回归（使用梯度上升法）"""
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(xArr,yArr):
    dataMat=mat(xArr)
    labelMat=mat(yArr).transpose()
    m,n=shape(dataMat)
    alpha=0.01
    maxCycles=500
    ws=ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMat*ws)
        error=(labelMat-h)
        ws=ws+alpha*dataMat.transpose()*error
    return ws

"""逻辑回归（使用牛顿法）"""
def logisticRegress(xArr,yArr):
    # 最大迭代次数
    max_iters = 40

    # xArr的size
    m=len(xArr)     # 训练集的个数
    n=len(xArr[0])  # 训练集特征维度

    xMat=mat(xArr);yMat=mat(yArr).T

    ws=zeros((n,1))
    ll=zeros((max_iters,1))

    for i in range(max_iters):
        margins=multiply((xMat*ws),yMat)
        ll[i] = (1.0 / m) * sum(log(1 + exp(-margins)))
        probs = 1.0 / (1 + exp(margins))
        grad = -(1.0 / m) * (xMat.T * (multiply(probs,yMat)))
        H = (1.0 / m) * (xMat.T * float(diag(multiply(probs,(1 - probs)))) * xMat)
        theta = ws - H.I*grad
        ws=theta[:,0]

    return ws

"""计算准确率"""
def calPrecision(xArr,ws,xValidationArr,yValidationArr):
    number=len(xValidationArr)
    right=0
    result=xValidationArr * ws
    #print xValidationArr*ws
    #print yValidationArr
    for i in range(number):
        if ((yValidationArr[0,i]==1.0)&(result[i,0]>1.0))|((yValidationArr[0,i]==2.0)&(result[i,0]<2.0)):
           right+=1
    if number==0:
        print 'This matrix is empty.'
    else:
        return float(right)/number

"""十折交叉验证"""
def tenFoldCrossValidation():
    xArray, yArray = loadDataSet('data/Titanic.txt', 'data/Titaniclabel.txt')
    xDividedArr, yDividedArr = divideDataSet(xArray, yArray)

    txtWrite = open('ten_fold_cross_validation.txt', 'w')

    for i in range(10):
        xValidationArr = []
        yValidationArr = []
        for k in range(len(xDividedArr[i])):
            xValidationArr.append(xDividedArr[i][k])
            yValidationArr.append(yDividedArr[i][k])
        xValidationArr = mat(xValidationArr)
        yValidationArr = mat(yValidationArr)
        for j in range(10):
            if i == j:    continue
            xArr = xDividedArr[j][0:]
            yArr = yDividedArr[j][0:]
        #ws = standRegress(xArr, yArr)
        #ws = logisticRegress(xArr, yArr)
        ws=gradAscent(xArr,yArr)
        xMat = mat(xArr)
        yMat = mat(yArr)
        precission = calPrecision(xMat, ws, xValidationArr, yValidationArr)
        txtWrite.write(str(precission))
        txtWrite.write('\n')

"""按比例验证"""
def ratioValidation():
    xArray, yArray = loadDataSet('data/Titanic.txt', 'data/Titaniclabel.txt')
    # 要取的比例
    percentList=[0.1,0.2,0.3,0.4,0.5]

    txtWrite = open('ration_validation.txt', 'w')

    for percent in percentList:
        xDividedArr, yDividedArr = divideRatioDataSet(xArray, yArray,percent)

        xValidationArr = []
        yValidationArr = []
        for k in range(len(xDividedArr[1])):
            xValidationArr.append(xDividedArr[1][k])
            yValidationArr.append(yDividedArr[1][k])
        xValidationArr = mat(xValidationArr)
        yValidationArr = mat(yValidationArr)

        xArr = xDividedArr[0][0:]
        yArr = yDividedArr[0][0:]
        #ws = standRegress(xArr, yArr)
        #ws = logisticRegress(xArr, yArr)
        ws = gradAscent(xArr, yArr)
        xMat = mat(xArr)
        yMat = mat(yArr)
        precission = calPrecision(xMat, ws, xValidationArr, yValidationArr)
        txtWrite.write(str(precission))
        txtWrite.write('\n')

#tenFoldCrossValidation()
ratioValidation()