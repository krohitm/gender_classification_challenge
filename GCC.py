from sklearn import svm
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score

#[height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

## Creating 3 classifiers
clf_svm = svm.SVC()
clf_KNN = neighbors.KNeighborsClassifier()
clf_NN = MLPClassifier()

#Training the classifiers on our data

clf_svm.fit(X, Y)
clf_KNN.fit(X, Y)
clf_NN.fit(X, Y)

#testing using the same data
pred_svm = clf_svm.predict(X)
acc_SVM = accuracy_score(Y, pred_svm)*100
print "Accuracy for SVM: {}".format(acc_SVM)

pred_KNN = clf_KNN.predict(X)
acc_KNN = accuracy_score(Y, pred_KNN)*100
print "Accuracy for KNN: {}".format(acc_KNN)

pred_NN = clf_NN.predict(X)
acc_NN = accuracy_score(Y, pred_NN)*100
print "Accuracy for NN: {}".format(acc_NN)

#choosing the most accurate classifier
index = np.argmax([acc_SVM, acc_KNN, acc_NN])
cls_dict = {0: 'SVM', 1: 'K Nearest Neighbors', 2: 'Neural Networks'}
print "The best gender classifier was: {}".format(cls_dict[index])
