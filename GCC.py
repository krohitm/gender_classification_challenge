from sklearn import svm
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



## Creating 3 classifiers
clf1 = svm.SVC()
clf2 = neighbors.KNeighborsClassifier()
clf3 = MLPClassifier()

#[height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']


#Training the classifiers on our data

clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)

#initializing accuracy score

score1 = 0
score2 = 0
score3 = 0

S=len(X)


prediction1 = []
prediction2 = []
prediction3 = []

for i in range(S):
    prediction1.append(clf1.predict(X[i])) #prediction from SVM
    if prediction1[i] == Y[i]:              #checking accuracy on training data itself
        score1 += 1

    prediction2.append(clf2.predict(X[i])) #prediction for K nearest neibghors
    if prediction2[i] == Y[i]:
        score2 += 1

    prediction3.append(clf3.predict(X[i])) #prediction for neural networks
    if prediction3[i] == Y[i]:
        score3 += 1

##printing the best model
scoreFinal = max(score1, score2, score3)
if scoreFinal == score1:
    print "Support Vector Machine"
elif scoreFinal == score2:
    print "Nearest Neighbors"
else:
    print "Neural Networks"
