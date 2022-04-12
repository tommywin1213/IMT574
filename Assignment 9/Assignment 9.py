# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 19:14:30 2022

@author: Tommy
"""

import pandas as pd
import keras 

dataset = pd.read_table('faults.nna')

X = pd.DataFrame(dataset.iloc[:,3:13].values)
y = dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X.loc[:,1] = labelencoder_X_1.fit_transform(X.iloc[:,1])

onehotencoder = OneHotEncoder(categories = 'auto')
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

from keras.models import Sequential
from keras.layers import Dense

nn = Sequential()
nn.add(Dense(11, activation = 'relu'))

nn.add(Dense(6, activation = 'relu'))
nn.add(Dense(1, activation = 'sigmoid'))

nn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

nn.fit(X_train, y_train)

y_pred = nn.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))