from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

iris = load_iris()

#max_depth is useless ??
clf = tree.DecisionTreeClassifier(max_depth=10, min_samples_leaf=9)

dataSet: np.ndarray = iris.data
labelSet: np.ndarray = iris.target

trainingLen: int = int(0.7*len(dataSet))

trainingSet = dataSet[:trainingLen]
trainingLabels = labelSet[:trainingLen]

testSet = dataSet[trainingLen:]
testLabels = labelSet[trainingLen:] 

print(trainingSet, testSet)

#training
clf = clf.fit(trainingSet, trainingLabels)

#testing
testResults = clf.predict(testSet)
print(testResults)

correctGuess = 0
wrongGuess = 0

for i in range(0, len(testResults)):
    if testResults[i] == testLabels[i]:
        correctGuess += 1
    else:
        wrongGuess += 1

print(correctGuess, wrongGuess)
print('Success rate : ', round((correctGuess / (correctGuess + wrongGuess)), 2))

with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
