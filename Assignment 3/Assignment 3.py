# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 15:50:26 2022

@author: Tommy
"""

import pandas as pd 
import numpy as np 
import scipy as scp
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics 
from sklearn.metrics import confusion_matrix

df = pd.read_csv('quality.csv', index_col=None)

import statsmodels.api as sm
import matplotlib.pyplot as plt

X = df[['bin_end_qmark', 'num_sentences']]
y = df['label']

print(list(X.columns.values)) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

print('LR CLassification report on test data')
print(classification_report(y_test, predictions))

print('LR Accuracy on test data')
print(accuracy_score(y_test, predictions))

print('LR Confusion matrix of test data')
print(confusion_matrix(y_test, predictions))


#Problem 2
df2 = pd.read_csv('wine.csv', index_col = None)

X = df2[['fixed_acidity', 'volatile_acidity', 'citric_acid', 'free_sulfur_dioxide']]
y = df2['high_quality']

data_top = df2.head()
print(data_top)

X_normalized = df2[['fixed_acidity', 'volatile_acidity', 'citric_acid', 'free_sulfur_dioxide']].apply(lambda x: (x -min(x))/(max(x)-min(x)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.70)

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

print('KNN Accuracy on test data')
print(accuracy_score(y_test, predictions))

print('KNN Confusion matrix of test data')
print(confusion_matrix(y_test, predictions))

