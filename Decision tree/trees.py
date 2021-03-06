#!/usr/bin/env python2
# -*- coding:utf-8 -*-

# 决策树

from math import log
from numpy import *

import operator,pickle

from treePlotter import createPlot

def createDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

"""计算给定数据集的香农熵"""
def calcShannonEnt(dataSet):
    # 计算数据集中实例的总数
    numEntries=len(dataSet)
    # 数据字典
    labelCounts={}
    # 计算数据熵的主要代码
    for featVec in dataSet:
        # 当前标签，键值是最后一列的数值
        currentLabel=featVec[-1]

        # 如果当前键值不存在，则扩展字典并将当前键值加入字典；
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 每一个键值都记录了当前类别出现的次数
        labelCounts[currentLabel] += 1

    shannonEnt=0.0
    for key in labelCounts:
        # 使用所有类标签的发生概率计算类别出现的概率
        prob=float(labelCounts[key])/numEntries
        # 计算香农熵
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

"""划分数据集"""
def splitDataSet(dataSet,axis,value):
    '''
    :param dataSet:待划分的数据集
    :param axis: 划分数据集的特征
    :param value: 需要返回的特征的值
    :return:
        splitDataSet(myDat,0,0)->[[1,'no'],[1,'no']]
        splitDataSet(myDat,0,1)->[[1,'yes'],[1,'yes'],[0,'no']]
    '''
    # 创建新的list对象
    retDataSet=[]
    for featVec in dataSet:
        # 抽取符合特征的数据
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            '''
            append与extend的区别:
            >>> a=[1,2,3]
            >>> b=[4,5,6]
            >>> a.append(b)
            >>> a
            >>> [1,2,3,[4,5,6]]
            >>> a.extend(b)
            >>> a
            >>> [1,2,3,4,5,6]
            '''
            retDataSet.append(reducedFeatVec)
    return retDataSet

"""选择最好的数据集划分方式"""
def chooseBestFeatureToSplit(dataSet):
    # 特征属性的数目(dataSet是由列表元素组成的列表，所有列表元素长度一致，且列表元素最后一个元素是类别标签)
    numFeatures=len(dataSet[0])-1
    # 原始香农熵，保存最初的无序度量值
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0.0; bestFeature=-1
    # 计算每种划分方式的信息熵
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        '''从列表中创建集合是得到列表中唯一元素值最快的方法'''
        uniqueVals=set(featList)
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        # 计算做好的信息熵（信息增益是熵的减少或者是数据无序度的减少）
        if (infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

"""投票表决（数据集已经处理了所有属性，但类标签依然不是唯一的情况下）"""
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    # operator.itemgetter(0),利用字典的key值进行排序;operator.itemgetter(1)，利用字典的value值进行排序
    # {'a':3,'b':8,'c':1}->[('b',8),('a',3),('c',1)]
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True) # reverse 递减排序
    return sortedClassCount[0][0]

"""创建树"""
def createTree(dataSet,labels):
    # 标签列表
    classList=[example[-1] for example in dataSet]
    # 类别完全相同则停止继续划分
    if classList.count(classList[0])==len(classList):
        return classList[0]
    # 遍历完所有的特征时返回出现次数最多的标签
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    # 选取最好的特征
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

"""使用决策树的分类函数"""
## 使用方法： classify(myTree, labels, [0,0]) -> 'no' ##
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    # 将标签字符串转化为索引
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:   classLabel = secondDict[key]
    return classLabel

# 将分类器存储到硬盘上
def storeTree(inputTree, filename):
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

# 读取硬盘上的分类器
def grabTree(filename):
    fr = open(filename, 'r')
    return pickle.load(fr)

"""构建lenses决策树"""
def lensesCreateTree():
    fr = open('lenses.txt','r')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    storeTree(lensesTree, 'lensesClassifier')

"""绘制决策树"""
def lensesShowTree():
    myTree = grabTree('lensesClassifier')
    createPlot(myTree)

"""lenses分类器"""
def lensesClassifier(testVec):
    myTree = grabTree('lensesClassifier')
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return classify(myTree, lensesLabels, testVec)

# lensesCreateTree()

# lensesShowTree()

# print lensesClassifier(['presbyopic','hyper','no','normal'])