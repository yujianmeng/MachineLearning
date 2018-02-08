from numpy import *
from math import log
import operator
import pickle

# ID3算法构造决策树

def createDataSet():
    dataSet=[[1, 1, 'yes'],
             [1, 1, 'yes'],
             [1, 0, 'no'],
             [0, 1, 'no'],
             [0, 1, 'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

dataSet,labels=createDataSet()

#计算香农熵
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCount={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel]=0
        labelCount[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCount:
        prob=float(labelCount[key])/numEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

print(calcShannonEnt(dataSet))

#根据特性划分数据集
#dataSet:需要决策的数据集
#axis:划分数据集的特征值
#value：划分数据特征值的当前值
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

print(splitDataSet(dataSet,0,1))
print(splitDataSet(dataSet,0,0))

#选择最好的数据集划分方式
def chooseBestFeatureToSlip(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if infoGain>bestInfoGain:
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

print(chooseBestFeatureToSlip(dataSet))

#多数表决
def majorityCnt(classList):
    classCount={}   #计数字典
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return classCount[0][0]

#创建决策树
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSlip(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}   #代表树结构信息的嵌套字典
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLables=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(
            dataSet, bestFeat, value), subLables)
    return myTree

tmpLabels = labels[:]
myTree=createTree(dataSet,tmpLabels)
print(myTree)

#根据构造的决策树分类器进行分类测试
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in list(secondDict.keys()):
        if key==testVec[featIndex]:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

classLabel = classify(myTree, labels, (1,1))
print(classLabel)

#将构造的决策树持久化到磁盘，随用随取
def storeTree (myTree, filename):
    fw = open(filename, 'wb')
    pickle.dump(myTree, fw)
    fw.close()

def grabTree (filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)

storeTree(myTree, '..\Resources\classierfierStorage.txt')
getTree = grabTree('..\Resources\classierfierStorage.txt')
print(getTree)