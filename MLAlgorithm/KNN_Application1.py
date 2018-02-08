from numpy import *
import operator
import matplotlib.pyplot as plt

#约会网站中对象分类
def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

dataMat,labels=file2matrix("E:\idea_workplace\MachineLearning\Resources\datingTestSet2.txt")

fig=plt.figure()
ax=fig.add_subplot(111) #将画布分割成1行1列，图像画在从左到右从上到下的第1块
ax.scatter(dataMat[:,0],dataMat[:,1],15.0*array(labels),15.0*array(labels))
plt.show()

#数据归一化
def autoNormal(dataSet):
    min=dataSet.min(0)  #参数0代表从列中选取最小值而不是行
    max=dataSet.max(0)
    ranges=max-min
    normalDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]  #shape读取矩阵的长度，比如shape[0]就是读取矩阵第一维度的长度
    normalDataSet=dataSet-tile(min,(m,1))
    normalDataSet=normalDataSet/tile(ranges,(m,1))
    return normalDataSet,ranges,min

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

def datingClassTest():
    hoRatio=0.1
    datingDataMat,datingLabels=file2matrix("E:\idea_workplace\MachineLearning\Resources\datingTestSet2.txt")
    normalMat,ranges,min=autoNormal(datingDataMat)
    m=normalMat.shape[0]
    numTestVec=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVec):
        classifierResult=classify0(normalMat[i,:],normalMat[numTestVec:m,:],
                                   datingLabels[numTestVec:m],3)
        print("the classifier came back with:%d,the real anwser is:%d"
              %(classifierResult,datingLabels[i]))
        if(classifierResult!=datingLabels[i]):
            errorCount+=1.0

    print("the total error rate is:%f"%(errorCount/float(numTestVec)))

def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(input("percentage of time spent playing video games?"))
    ffMiles=float(input("freequent flier miles earned per year?"))
    iceCream=float(input("liters of ice cream consumed per yaer?"))
    datingDataMat, datingLabels = file2matrix("E:\idea_workplace\MachineLearning\Resources\datingTestSet2.txt")
    normalMat,ranges,mins = autoNormal(datingDataMat)
    inArr=array([ffMiles,percentTats,iceCream])
    classifierResult=classify0((inArr-mins)/ranges,normalMat,datingLabels,3)
    print("You will probably like this person:",resultList[classifierResult-1])

classifyPerson()