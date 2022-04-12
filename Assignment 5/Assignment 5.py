# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 15:30:12 2022

@author: Tommy
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch

data = pd.read_csv('airline-safety.csv')

X = data.iloc[:, [2, 7]]
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
plt.scatter(data['incidents_00_14'], data['fatalities_00_14'], c=y_means, cmap='rainbow')
plt.scatter(centroids[:,0], centroids[:,1], c='black',s=100)
plt.show()

plt.figure(figsize=(10,10))
plt.title('Agglomerative clustering')
Dendrogram = sch.dendrogram((sch.linkage(X, method='ward')))

ac = AgglomerativeClustering(n_clusters = 4)
y_ac = ac.fit_predict(X)

print('Aggomerative clustering assignments', y_ac)