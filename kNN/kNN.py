#!/usr/bin/env python2
#-*- coding:utf-8 -*-

# k-近邻算法

from numpy import *
from os import listdir
import operator
from PIL import Image

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

"""将文本记录转换为该分类器能够解析的格式"""
def file2matrix(filename):
    # 打开文件
    fr=open(filename)
    arrayOLines=fr.readlines()
    # 得到文件行数,文本中每行为一个样本数据
    numberOfLines=len(arrayOLines)
    # 创建返回的NumPy矩阵,以零填充
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    # 解析文件数据到列表
    for line in arrayOLines:
        # 截掉所有的回车字符
        line=line.strip()
        # 使用tab字符\t将上一步得到的整行数据分割为一个元素列表
        listFromLine=line.split('\t')
        # 选取元素列表中的前3个元素，将它们存储到特征矩阵中
        returnMat[index,:]=listFromLine[0:3]
        # 使用索引值-1表示列表中的最后一列元素，最后一列元素是标签值，存储到向量classLabelVector
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

"""将图像转换为测试向量"""
def img2vector(filename):
    # 1行32*32列的向量
    returnVect=zeros((1,1024))
    fr=open(filename)
    # 将图像（32*32）转换为1*1024的向量
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*j+i]=int(lineStr[i])
    # 返回returnVect
    return returnVect

"""归一化特征值"""
def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    # 特征值相除
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

"""kNN核心算法"""
def classify0(inX,dataSet,labels,k):
    '''计算距离'''
    # 获取dataset的维度
    datasetSetSize=dataSet.shape[0]
    # tile函数：tile(list,(x,y))，在 a 维的array中，每一维度都重复list b 次
    diffMat=tile(inX,(datasetSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    # sum(axis=0):[[a1,a2],[b1,b2]]->[a1+b1,a2+b2];sum(axis=1):[[a1,a2],[b1,b2]]->[a1+a2,b1+b2]
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5

    # argsort函数返回的是数组值从小到大的索引值
    sortedDistIndices=distances.argsort()
    classCount={}

    '''选择距离最小的k个点'''
    for i in range(k):
        voteIlabel=labels[sortedDistIndices[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1

    '''排序'''
    # key:指出用于排序的键值    reverse:真为降序，假为升序
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)

    return sortedClassCount[0][0]

"""分类器的测试代码"""
def datingClassTest():
    # 用来测试的数据集占总的数据集的比率
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('kNN/datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    # 总的数据集的数目
    m=normMat.shape[0]
    # 用于测试的数据集的数目
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    # 测试kNN分类器
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with :%d,the real answer is :%d"%(classifierResult,datingLabels[i])
        if classifierResult!=datingLabels[i]:   errorCount+=1.0
    print "the total error rate is :%f"%(errorCount/float(numTestVecs))

"""手写数字识别系统的测试代码"""
def handwritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('kNN/digits/trainingDigits')
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('kNN/digits/trainingDigits/%s'%fileNameStr)
    testFileList=listdir('kNN/digits/testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileNameStr.split('_')[0])
        vectorUnderTest=img2vector('kNN/digits/testDigits/%s'%fileNameStr)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print "the classifier came back with :%d,the real answer is :%d"%(classifierResult,classNumStr)
        if (classifierResult!=classNumStr):     errorCount+=1.0
    print "\nthe total number of errors is:%d"%errorCount
    print "\nthe total error rate is:%f"%(errorCount/float(mTest))

"""预测函数"""
def classifyPerson():
    # 三种标签
    resultList=['not at all','in small doses','in large doses']
    # 用户输入
    percentTats=float(raw_input('percentage of time spent playing video games?'))
    ffMiles=float(raw_input('frequent flier miles earned per year?'))
    iceCream=float(raw_input('liters of ice cream consumed per year?'))
    # 数据处理
    datingDataMat,datingLabels=file2matrix('kNN/datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([percentTats,ffMiles,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "You will probably like this person:",resultList[classifierResult-1]

"""手写数字识别系统的预测函数"""
def classifyNumber(inX):
    hwLabels=[]
    trainingFileList=listdir('kNN/digits/trainingDigits')
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('kNN/digits/trainingDigits/%s'%fileNameStr)
    classifierResult=classify0(inX,trainingMat,hwLabels,5)
    print classifierResult


"""
'''for visual modeling'''
datingDataMat,datingLabels=file2matrix('kNN/datingTestSet2.txt')

normMat,ranges,minVals=autoNorm(datingDataMat)
print(normMat)
print(ranges)
print(minVals)

import matplotlib
import matplotlib.pyplot as plt

fig=plt.figure()
# 111的意思：将画布分成1列1行，用从左到右从上到下的第一个来绘图
ax=fig.add_subplot(111)
# datingDateMat[:,0]表示使用datingDateMat矩阵的第一列数据，其他类推
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
plt.show()
"""

"""
'''for datingClassTest'''
datingClassTest()
"""

"""
'''for classifyPerson'''
classifyPerson()
"""

"""
'''for handwritingClassTest'''
handwritingClassTest()
"""

"""
'''for classifyNumber'''
img=Image.open('C:\\Users\\Kingboung\\Desktop\\4.jpg','r')
inX=[]
for y in range(32):
    for x in range(32):
        if img.getpixel((x,y))==(255,255,255): inX.append(0)
        else: inX.append(1)
print inX
classifyNumber(inX)
img.close()
"""