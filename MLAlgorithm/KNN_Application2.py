from numpy import *
import operator
from os import listdir

#手写识别系统
def classify0(inx,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inx,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()            #argsort函数返回的是数组值从小到大的索引值
    classCount={}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

def handWritingClassTest():
    hwLabels=[]
    trainingFileList=listdir("E:\idea_workplace\MachineLearning\Resources\\trainingDigits")
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('E:\idea_workplace\MachineLearning\Resources\\trainingDigits\%s' %fileNameStr)
    testFileList=listdir('E:\idea_workplace\MachineLearning\Resources\\testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('E:\idea_workplace\MachineLearning\Resources\\testDigits\%s' %fileNameStr)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with:%d,the real anwser is:%d " %(classifierResult,classNumStr))
        if(classifierResult!=classNumStr):
            errorCount+=1
    print("the total number of errors is:%d" %errorCount)
    print("the total error rate is:%f" %(errorCount/float(mTest)))

handWritingClassTest()