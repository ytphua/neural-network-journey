# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 01:26:13 2017
original source: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/linear_regression_class
@author: tsann
"""
import numpy as np
import matplotlib.pyplot as plt

#load data
X = []
Y = []

for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

#convert X & Y into numpy arrays
X = np.array(X)
Y = np.array(Y)

#plot data
plt.scatter(X, Y)
plt.show()

#calculate
denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean()*X.sum())/denominator 
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y))/denominator 
Yhat = a*X + b

#plot data & result
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

#computer r-square
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1)/d2.dot(d2)
print("R2 = "+str(r2))
