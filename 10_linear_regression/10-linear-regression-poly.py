# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 02:22:14 2017
original source: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/linear_regression_class
@author: tsann
"""

import re
import numpy as np
import matplotlib.pyplot as plt

#loadd data
X = []
Y = []

non_decimal = re.compile(r'[^\d]+')

for line in open('data_poly.csv'):
    x, y = line.split(',')
    x = float(x)
    X.append([1, x, x*x])
    Y.append(float(y))

#convert X & Y into numpy arrays
X = np.array(X)
Y = np.array(Y)

#plot data
plt.scatter(X[:,1], Y)
plt.show()

#calculate weight
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T,Y))
Yhat = np.dot(X, w)

#plot all data
plt.scatter(X[:,1],Y)
plt.plot(sorted(X[:,1]),sorted(Yhat))
plt.show()

##computer r-square
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1)/d2.dot(d2)
print("R2 = "+str(r2))

