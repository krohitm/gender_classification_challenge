from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

clf = tree.DecisionTreeClassifier()

## CHALLENGE - create 3 more classifiers...
clf1 = svm.SVC()
clf2 = neighbors.KNeighborsClassifier()
clf3 = MLPClassifier()

#[height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']


#CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)

#initializing accuracy score
score = 0
score1 = 0
score2 = 0
score3 = 0

S=len(X)

prediction = []
prediction1 = []
prediction2 = []
prediction3 = []

for i in range(S):
    prediction.append(clf.predict(X[i]))   #prediction from Decision trees
    if prediction[i] == Y[i]:           #checking accouracy
        score += 1

    prediction1.append(clf1.predict(X[i])) #prediction from SVM
    if prediction1[i] == Y[i]:
        score1 += 1

    prediction2.append(clf2.predict(X[i])) #prediction for nearest neibghors
    if prediction2[i] == Y[i]:
        score2 += 1

    prediction3.append(clf3.predict(X[i])) #prediction for neural networks
    if prediction3[i] == Y[i]:
        score3 += 1

scoreFinal = max(score, score1, score2, score3)
if scoreFinal == score:
    print "Decision Trees"
elif scoreFinal == score1:
    print "Support Vector Machine"
elif scoreFinal == score2:
    print "Nearest Neighbors"
else:
    print "Neural Networks"

#prediction = clf.predict([[190, 70, 43]])
#prediction1 = clf1.predict([[190, 70, 43]])
#prediction2 = clf2.predict([[190, 70, 43]])
#prediction3 = clf3.predict([[190,70,43]])
#CHALLENGE compare their reusults and print the best one!

#print prediction
#print prediction1
#print prediction2
#print prediction3
