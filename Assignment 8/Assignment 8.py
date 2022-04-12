# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 19:22:11 2022

@author: Tommy
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher

ships = pd.read_csv('ships.csv')

ships = ships.dropna()

X = ships.iloc[:, [2,3,4]]
y = ships.iloc[:, [6]]

X = pd.get_dummies(X)

#SVM based model
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#model = SVC(kernel = 'poly', degree = 1)
#model.fit(X_train, y_train)

#predictions = model.predict(X_test)

#print(accuracy_score(y_test, predictions))
#print(confusion_matrix(y_test, predictions))
#print(classification_report(y_test, predictions))

#Logistic Regression
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

#logmodel = LogisticRegression()

#logmodel.fit(X_train, y_train)

#predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

#print('LR CLassification report on test data')
#print(classification_report(y_test, predictions))

#print('LR Accuracy on test data')
#print(accuracy_score(y_test, predictions))

#print('LR Confusion matrix of test data')
#print(confusion_matrix(y_test, predictions))

#Random Forest Model
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))

