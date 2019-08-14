#Basic implementation of knn
import sklearn
from sklearn.datasets import load_iris
from numpy import array
from scipy.spatial import distance
def euc(a,b):
    return distance.euclidean(a,b)
    
class classify():
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    def closest(self, row):
        best_dist = euc(row, self.x_train[0])
        best_index = 0
        for i in range(1, len(self.x_train)):
            dist = euc(row,x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]
iris = load_iris()
x = iris.data
y = iris.target
    
#splitting data to train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)
#Decision tree approach
clf = classify()
clf.fit(x_train, y_train)
    
    #Finding the accuracy of the classifier
    
predictions = clf.predict(x_test)
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
    
