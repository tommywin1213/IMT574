# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 21:42:37 2022

@author: Tommy
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np

#Problem 1

derm = pd.read_csv('dermatology.csv', delimiter = '\t')
derm = derm.apply(pd.to_numeric, errors = 'coerce')
derm = derm.dropna()

#Gradient Descent Model
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as nm


y = derm.Disease
X = derm[['Age']]
X = sm.add_constant(X)
lr_model = sm.OLS(y,X).fit()
print("_____________Printing the model Summary____________")
print(lr_model.summary())
print("____________Printing the model's parameters________")
print(lr_model.params)

plt.figure()
plt.scatter(derm.Age, derm.Disease)
plt.xlabel('Age')
plt.ylabel('Disease')

from mpl_toolkits.mplot3d import Axes3D

X_axis, Y_axis = np.meshgrid(np.linspace(X.Age.min(), X.Age.max(), 100), np.linspace(X.Age.min(), X.Age.max(), 100))

Z_axis = lr_model.params[0] + lr_model.params[1] * Y_axis

fig = plt.figure(figsize=(12,8))
ax = Axes3D(fig, azim=-100)

ax.plot_surface(X_axis, Y_axis, Z_axis, cmap=plt.cm.coolwarm, alpha=0.5, linewidth = 0)
ax.scatter(X.Age, X.Age, y)
ax.set_xlabel('Age')
ax.set_zlabel('Disease')

#Random Forest Model
print('Random Forest Model')
X = derm.iloc[: , [0, 33]]
y = derm['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))

#kNN Model
from sklearn.neighbors import KNeighborsClassifier

X_normalized = derm.iloc[:,[0,33]].apply(lambda x: (x -min(x))/(max(x)-min(x)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.70)

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

print('KNN Accuracy on test data')
print(accuracy_score(y_test, predictions))

print('KNN Confusion matrix of test data')
print(confusion_matrix(y_test, predictions))

#Clustering
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch


X = derm.iloc[:, [0, 32]]
scaler = StandardScaler()
scaler.fit_transform(X)

kmeans = KMeans(n_clusters = 4)
y_means = kmeans.fit_predict(X)

centroids = kmeans.cluster_centers_
print(centroids)
print('kmeans clustering assignments', y_means)

# Visualize the clusters
plt.figure(figsize=(10,10))
plt.title('Divisive clustering with k-means')
plt.scatter(derm['Age'], derm['Disease'], c=y_means, cmap='rainbow')
plt.scatter(centroids[:,0], centroids[:,1], c='black',s=100)
plt.show()

plt.figure(figsize=(10,10))
plt.title('Agglomerative clustering')
Dendrogram = sch.dendrogram((sch.linkage(X, method='ward')))

ac = AgglomerativeClustering(n_clusters = 4)
y_ac = ac.fit_predict(X)

print('Aggomerative clustering assignments', y_ac)

#Problem 2 
import matplotlib.pyplot as plt 
from statsmodels import api as sm

crimes = pd.read_csv('hatecrime.csv')

#Remove all NaN values
crimes = crimes.dropna()

#Income inequality vs hate crimes 
y= crimes.iloc[:, 11]
X= crimes.iloc[:,1]
X = sm.add_constant(X)
lr_model = sm.OLS(y, X).fit()
print(lr_model.summary())
print(lr_model.params)

#Hate crime predictor
x = crimes.iloc[:,[8,9]]
y = crimes.iloc[:,11]

x = sm.add_constant(x) 
 
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
 
print_model = model.summary()
print(print_model)

#Variation in Hate Crime
X = crimes.iloc[:, [1, 11]]
scaler = StandardScaler()
scaler.fit_transform(X)

kmeans = KMeans(n_clusters = 4)
y_means = kmeans.fit_predict(X)

centroids = kmeans.cluster_centers_
print(centroids)
print('kmeans clustering assignments', y_means)

# Visualize the clusters
plt.figure(figsize=(10,10))
plt.title('Divisive clustering with k-means')
plt.scatter(crimes['hate_crimes_per_100k_splc'], crimes['avg_hatecrimes_per_100k_fbi'], c=y_means, cmap='rainbow')
plt.scatter(centroids[:,0], centroids[:,1], c='black',s=100)
plt.show()

plt.figure(figsize=(10,10))
plt.title('Agglomerative clustering')
Dendrogram = sch.dendrogram((sch.linkage(X, method='ward')))

ac = AgglomerativeClustering(n_clusters = 8)
y_ac = ac.fit_predict(X)

print('Aggomerative clustering assignments', y_ac)

