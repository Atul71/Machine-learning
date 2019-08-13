import sklearn
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
from numpy import array
iris = load_iris()
#print (iris.feature_names)
#print (iris.target_names)
#print (iris.data[0])

X = iris.data
y = iris.target

#splitting data to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

#Decision tree approach
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

#Finding the accuracy of the classifier

predictions = clf.predict(X_test)
print("Accuracy of algorithm is")
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, predictions))

print("Enter the features of flower to be predicted")
a=[]
for i in range(4):
    print(iris.feature_names[i])
    x=float(input())
    a.append(x)
predict_data = np.array(a)
predict_data=predict_data.reshape(1,-1)
print (iris.target_names[clf.predict(predict_data)])
