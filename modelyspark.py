from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext 
sc=SparkContext()

from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils

from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

def line(l):
    diction = {" False." : 0, " True." : 1, " no" : 0, " yes" : 1}
    pole = [x+6 for x in range(14)]
    for i in [1,2]:
        pole.append(i)
    row = l.split(",")
    for i in pole:
        row[i] = float(row[i])        
    row[20] = diction[row[20]]
    row[4] = diction[row[4]]
    row[5] = diction[row[5]]
    return LabeledPoint(row[20], row[1:3] + row[4:20])


datatren = sc.textFile("/home/katka/Desktop/churn/tren.txt")
tren = datatren.map(line)
datatest = sc.textFile("/home/katka/Desktop/churn/test.txt")
test = datatest.map(line)

def predictmodel(model, test):
    predictions = model.predict(test.map(lambda p: p.features))
    labelsAndPreds = test.map(lambda p: p.label).zip(predictions)
    trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count()/float(test.count())
    print("Training Error = " + str(trainErr))
    print("Pocet nezhod = " + str(labelsAndPreds.filter(lambda (v, p): v != p).count()))


def logregression(tren, test):
    modelreg = LogisticRegressionWithLBFGS.train(tren)
    predictmodel(modelreg, test)


def destree(tren, test):
    modeltree = DecisionTree.trainClassifier(tren, numClasses=2, categoricalFeaturesInfo= {}, impurity='gini', minInstancesPerNode=20)
    predictmodel(modeltree, test)


def ranforest(tren, test):
    modelles = RandomForest.trainClassifier(tren, numClasses=2, categoricalFeaturesInfo={}, numTrees=100, featureSubsetStrategy="auto", impurity='gini', maxDepth=10)
    predictmodel(modelles, test)




logregression(tren, test)
destree(tren, test)
ranforest(tren, test)