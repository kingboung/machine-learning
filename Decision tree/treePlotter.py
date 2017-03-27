#!/usr/bin/env python2
# -*- coding:utf-8 -*-

from pylab import *
# 使汉字正常显示
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

"""获取叶节点的数目"""
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs += 1
    return numLeafs

"""获取树的层数"""
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth:    maxDepth = thisDepth
    return maxDepth

"""绘制带箭头的注解"""
def plotNode(nodeText, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeText,
                            xy=parentPt,
                            xycoords='axes fraction',
                            xytext=centerPt,
                            textcoords='axes fraction',
                            va="center", ha="center",
                            bbox=nodeType,
                            arrowprops=arrow_args)

"""在父子节点间填充文本信息"""
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

"""绘制树形图"""
def plotTree(myTree, parentPt, nodeTxt):
    # 计算宽和高
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
        # 存储树的宽度
        plotTree.totalW = float(getNumLeafs(inTree))
        # 存储树的高度
        plotTree.totalD = float(getTreeDepth(inTree))
        # xOff和yOff用于追踪已经绘制的节点位置（x轴：0.0~1.0；y轴：0.0~1.0）
        plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0
        plotTree(inTree, (0.5,1.0), '')
        plt.show()

# def createPlot():
#     fig = plt.figure(1, facecolor='blask')
#     fig.clf()
#     createPlot.ax1 = plt.subplot(111, frameon=False)
#     plotNode(U'决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
#     plotNode(U'叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
#     plt.show()

"""输出预先存储的树信息，避免每次测试代码都要从数据中创建的麻烦"""
def retrieveTree(i):
    listOfTrees = [
        {'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},
        {'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}
    ]
    return listOfTrees[i]

# myTree=retrieveTree(1)
#
# print getNumLeafs(myTree)
# print getTreeDepth(myTree)
#
# createPlot(myTree)