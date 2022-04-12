# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:30:03 2022

@author: Tommy
"""

import pandas as pd 
import numpy as np 
import scipy as scp
import sklearn as sk
from scipy.sparse import csr_matrix
import sklearn as sk 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher

import statsmodels.api as sm
import matplotlib.pyplot as plt

#Logistic Regression
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Subset that contains features that we believe are relevant 
df = train.iloc[:, [1, 2, 3, 4, 5, 6, 10, 11, 23, 24, 25, 26, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40 , 41, 42, 43, 44]]

#DF for logistic regression 
lr_df = train.iloc[:, [2, 3, 4, 5, 36, 37, 43, 44]]

df2 = df.iloc[:, 0 : 27]
df2 = df2.dropna()

X = df2.iloc[:, 0 : 26]
y = df2['Genetic Disorder']

#Drops rows that have N/A values
X = X.dropna()
y = y.fillna('Not Available')

#Gets dummy variables so we are able to run logisitc regresion
X = pd.get_dummies(X)

#Logistic Regression 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
logmodel = LogisticRegression(max_iter=1000)
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

#Clustering
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch

X = lr_df
#Drops y values from x value dataset 
X = X.drop(columns = ['Genetic Disorder', 'Disorder Subclass'])

#Fill NA values 
X = X.fillna(0)
y = lr_df[['Genetic Disorder', 'Disorder Subclass']]
y = y.fillna('Not Available')

X = pd.concat([X, y], axis=1)

#Gets dummy variables for clustering 
X = pd.get_dummies(X)

#Agglomerative Cluster
plt.figure(figsize=(10,10))
plt.title('Agglomerative clustering')
Dendrogram = sch.dendrogram((sch.linkage(X, method='ward')))

ac = AgglomerativeClustering(n_clusters = 4)
y_ac = ac.fit_predict(X)

print('Agglomerative clustering assignments', y_ac)

#Random Forest Classifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import seaborn as sns 

#Plug in features of interest
X = train.iloc[:, 16]
y = train[['Genetic Disorder']]

#X = pd.get_dummies(X)

#Runs Random Forest Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

#Prints accuracy score and confusion matrix
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))


#Random Forest Classifier with Simple Imputer

#forloop generates data on how many na values there are
for i in range(len(lr_df.columns)):
    missing_data = lr_df[lr_df.columns[i]].isna().sum()
    perc = missing_data / len(lr_df) * 100
    print('>%d,  missing entries: %d, percentage %.2f' % (i, missing_data, perc))

#plots the n/a values 
plt.figure(figsize = (10,6))
sns.heatmap(lr_df.isna(), cbar = False, cmap = 'viridis', yticklabels = False);

#Simple Imputer using means 
imputer = SimpleImputer(strategy = 'mean')
model = RandomForestClassifier()

#Get dummy variables for X 
X = pd.get_dummies(X)
data = X.values

X = data[:, :-1]
y = data[:, -1]

#Calls simple imputer
imputer.fit(X)
X_trans = imputer.transform(X)

'Missing: {}'.format(sum(np.isnan(X).flatten()))
'Missing: {}'.format(sum(np.isnan(X_trans).flatten()))

#Random Forest Classifier with Simple Imputer
model = RandomForestClassifier()
pipeline = Pipeline([('impute', imputer), ('model', model)])
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

print('Mean Accuracy:{} std:{}'.format(round(np.mean(scores), 3), round(np.std(scores), 3)))

