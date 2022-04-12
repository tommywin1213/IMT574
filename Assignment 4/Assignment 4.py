# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 17:18:39 2022

@author: Tommy
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

hand = pd.read_csv('hand.csv')

X = hand[['handPost', 'thumbSty', 'region']]

y = hand[['post1906']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))

