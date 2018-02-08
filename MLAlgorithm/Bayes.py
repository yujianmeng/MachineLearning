# @Time    : 2018/1/17 12:04
# @Author  : yujian
# @File    : Bayes.py
# @Software: PyCharm
from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList, classVec

#dataSet, classVec = loadDataSet()
#构建字符集
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet|set(document)   #求并集
    return list(vocabSet)

#转换成词向量
def setOfWords2Vec(vocabSet, inputSet):
    returnVec = [0]*len(vocabSet)
    for word in inputSet:
        if word in vocabSet:
            returnVec[vocabSet.index(word)] = 1 #词集模型，若用词袋模型直接累加
        else:
            print("the word %s is not in my Vocabulary!" %word)
    return returnVec

#vocabSet = createVocabList(dataSet)
#print(vocabSet)
#wordVec = setOfWords2Vec(vocabSet, dataSet[0])
#print(wordVec)

#朴素贝叶斯分类器训练
def trainNB(trainMat, trainCategory):
    numTrainDocs = len(trainMat)
    numWords = len(trainMat[0])
    p1busive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 3.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p1Vect, p0Vect, p1busive

#朴素贝叶斯分类函数
def classifyNB(vec2Classify, p1busive, p0Vect, p1Vect):
    p0 = sum(vec2Classify*p0Vect)+log(1-p1busive)
    p1 = sum(vec2Classify*p1Vect)+log(p1busive)
    if p0 > p1:
        return 0
    else:
        return 1

def testNB():
    dataSet, classVec = loadDataSet()
    vocabSet = createVocabList(dataSet)
    trainMat = []
    for doc in dataSet:
        trainMat.append(setOfWords2Vec(vocabSet, doc))
    p1Vect, p0Vect, p1busive = trainNB(trainMat, classVec)
    testEntry = ['love', 'my', 'dalmation']
    testDocVec = setOfWords2Vec(vocabSet, testEntry)
    print(testEntry,"classified as:", classifyNB(testDocVec, p1busive, p0Vect, p1Vect))
    testEntry = ['stupid', 'garbage']
    testDocVec = setOfWords2Vec(vocabSet, testEntry)
    print(testEntry, "classified as:", classifyNB(testDocVec, p1busive, p0Vect, p1Vect))

#testNB()