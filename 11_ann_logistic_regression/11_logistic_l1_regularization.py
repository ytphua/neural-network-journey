# -*- coding: utf-8 -*-
"""
Created on Wed May  3 02:36:39 2017

@author: tsann
original source: https://github.com/lazyprogrammer/machine_learning_examples/blob/master/ann_logistic_extra/ecommerce_data.csv
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1 + np.exp(-z))

N = 50
D = 50

X = (np.random.random((N,D)) - 0.5)*10

true_w = np.array([1, 0.5, -0.5] + [0]*(D-3))

Y = np.round(sigmoid(X.dot(true_w) + np.random.randn(N)*0.5))

costs = []
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001
l1 = 2.0

for t in range(5000):
    Yhat = sigmoid(X.dot(w))
    delta = Yhat - Y
    w = w - learning_rate * (X.T.dot(delta) + l1*np.sign(w))
    
    cost = -(Y*np.log(Yhat) + (1-Y)*np.log(Yhat)).mean() + l1*np.abs(w).mean()
    costs.append(cost)

plt.plot(costs)
plt.show()

plt.plot(true_w, label='true_w')
plt.plot(w, label='w map')
plt.legend()
plt.show()