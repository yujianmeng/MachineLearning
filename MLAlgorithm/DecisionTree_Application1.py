from MLAlgorithm import DecisionTree
from MLAlgorithm import TreePlotter

#使用决策树预测隐形眼镜类型

def getDataSet(filename):
    lenses = []
    fr = open(filename)
    for inst in fr.readlines():
        lenses.append(inst.strip().split('\t'))
    return lenses

dataSet = getDataSet('..\Resources\lenses.txt')
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
tmpLabels = lensesLabels[:]
lensesTree = DecisionTree.createTree(dataSet, tmpLabels)
TreePlotter.createPlot(lensesTree)
result = DecisionTree.classify(lensesTree, lensesLabels, ('young', 'hyper', 'yes', 'normal'))
print(result)