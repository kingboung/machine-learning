#!usr/bin/env python2
# -*- coding:utf-8 -*-

# 决策树

from numpy import *
from math import log

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