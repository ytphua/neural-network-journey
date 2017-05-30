# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 02:22:14 2017
original source: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/linear_regression_class
@author: tsann
"""

import re
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#loadd data
X = []
Y = []

non_decimal = re.compile(r'[^\d]+')

for line in open('data_2d.csv'):
    x1, x2, y = line.split(',')
    X.append([1, float(x1), float(x2)])
    Y.append(float(y))

#convert X & Y into numpy arrays
X = np.array(X)
Y = np.array(Y)

#plot data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1], Y)
plt.show()

#calculate weight
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T,Y))
Yhat = np.dot(X, w)

##computer r-square
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1)/d2.dot(d2)
print("R2 = "+str(r2))

