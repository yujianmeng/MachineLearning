# @Time    : 2018/1/17 12:52
# @Author  : yujian
# @File    : Bayes_Application1.py
# @Software: PyCharm
#朴素贝叶斯过滤垃圾邮件
import re
from MLAlgorithm import Bayes
from numpy import *

def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [token.lower() for token in listOfTokens if len(token) > 2]

def classifyEmailTest():
    docList=[]
    classList=[]
    fullText = []
    for i in range(1, 26):
        bigString = open('..\Resources\email\spam\%d.txt' %i).read()
        wordList = textParse(bigString)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('..\Resources\email\ham\%d.txt' %i, encoding='utf-8').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocab = Bayes.createVocabList(docList)
    testSetIndex = []
    traingSetIndex = list(range(50))
    for i in range(10):
        randIndex = int(random.uniform(0,len(traingSetIndex)))
        testSetIndex.append(traingSetIndex[randIndex])
        del(traingSetIndex[randIndex])
    trainMat = []
    trainClass = []
    for docIndex in traingSetIndex:
        trainMat.append(Bayes.setOfWords2Vec(vocab, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p1Vect, p0Vect, p1busive = Bayes.trainNB(trainMat, trainClass)
    errorCount = 0
    for testDocIndex in testSetIndex:
        wordVect = Bayes.setOfWords2Vec(vocab, docList[testDocIndex])
        classifyResult = Bayes.classifyNB(wordVect, p1busive, p0Vect,p1Vect)
        if classifyResult != classList[testDocIndex]:
            errorCount += 1
            print(docList[testDocIndex], '\n')
    print("the error rate is:",float(errorCount)/len(testSetIndex))

classifyEmailTest()