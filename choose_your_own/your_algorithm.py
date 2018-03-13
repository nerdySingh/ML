#!/usr/bin/python

def naiveBayes(features_train, labels_train):
    from sklearn.naive_bayes import GaussianNB
    
    clf = GaussianNB()
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)

    acc = accuracy_score(pred,labels_test)
    
    
    return clf,acc




    pass

def supportVectorMachines(features_train,labels_train):
    from sklearn import svm
    clf = svm.SVC(C = 100.0 , gamma=100.0)
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)

    acc = accuracy_score(pred , labels_test)

    return clf,acc
    pass

def decisionTrees(features_train,features_test,labels_test,labels_train):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split = 2)
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)
    acc  = accuracy_score(pred,labels_test)
    return clf,acc
    
    pass

def knn(features_train,fetaures_test,labels_test,labels_train):
    from sklearn.neighbors import  KNeighborsClassifier

    clf = KNeighborsClassifier(n_neighbors=20,algorithm='brute',p=2)

    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)
    acc = accuracy_score(pred,labels_test)

    return clf,acc
    pass

def adabooost(features_train,features_test,labels_test,labels_train):
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier()
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)

    acc = accuracy_score(pred,labels_test)

    return clf,acc
    pass

def randomForest(features_train,features_test,labels_test,labels_train): 
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_features='sqrt',max_depth=25) 

    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)

    acc = accuracy_score(pred,labels_test)
    return clf,acc
    pass


import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
clf_final = []
clf_final.append(naiveBayes(features_train, labels_train))
clf_final.append(supportVectorMachines(features_train, labels_train))
clf_final.append(decisionTrees(features_train,features_test,labels_test,labels_train))
clf_final.append(knn(features_train,features_test,labels_test,labels_train))
clf_final.append(adabooost(features_train,features_test,labels_test,labels_train))
clf_final.append(randomForest(features_train,features_test,labels_test,labels_train))


clf_final = sorted(clf_final,key=lambda x:x[1],reverse=True)
print(clf_final[0][0])
clf = clf_final[0][0]





#clf_svm, acc_svm = SVM(features_train,features_test)







try:
    #print("Dude")
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass


