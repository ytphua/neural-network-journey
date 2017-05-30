# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 01:16:33 2017

@author: tsann
"""
# demonstration of L2 regularization
#
# notes for this course can be found at:
# https://deeplearningcourses.com/c/data-science-linear-regression-in-python
# https://www.udemy.com/data-science-linear-regression-in-python

import numpy as np
import matplotlib.pyplot as plt

N = 10
D = 3
X = np.zeros((N,D))
X[:,0] = 1
X[:5,1] = 1
X[5:,2] = 1
print(X)
Y = np.array([0]*5+[1]*5)
print(Y)
w=np.linalg.solve(np.dot(X.T, X), np.dot(X.T,Y))

costs = []
w = np.random.randn(D)/np.sqrt(D)
learning_rate = 0.001
for t in range(1000):
    Yhat = X.dot(w)
    delta = Yhat - Y
    w = w - learning_rate * X.T.dot(delta)
    mse = delta.dot(delta) / N
    costs.append(mse)

plt.plot(delta)
plt.show()