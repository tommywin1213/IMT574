# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 19:06:02 2022

@author: Tommy Huynh
"""

import pandas as pd 
import matplotlib.pyplot as plt 
from statsmodels import api as sm

#Problem 1

#Part A

df = pd.read_csv('airlines.csv', index_col=None)

x = df.iloc[:,[1,3]]
y = df.iloc[:,4]

x = sm.add_constant(x) 
 
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
 
print_model = model.summary()
print(print_model)

#Part B
x2 = df.iloc[:,4]
y2 = df.iloc[:,9]

x2 = sm.add_constant(x) 
 
model2 = sm.OLS(y2, x2).fit()
predictions2 = model.predict(x2) 
 
print_model2 = model2.summary()
print(print_model2)

#Problem 2
df2 = pd.read_csv('kangaroo.csv', index_col=None)

y= df2.iloc[:,1]
X= df2.iloc[:,0]
X = sm.add_constant(X)
lr_model = sm.OLS(y, X).fit()
print(lr_model.summary())
print(lr_model.params)

plt.figure()
plt.scatter(df2.iloc[:,1], df2.iloc[:,0])
plt.xlabel('Length')
plt.ylabel('Width')

m = 0
c = 0

L = 0.0001  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

# Performing Gradient Descent 
for i in range(epochs): 
    Y_pred = m*X + c  # The current predicted value of Y
    D_m = (-2/n) * sum(X * (y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    
print (m, c)