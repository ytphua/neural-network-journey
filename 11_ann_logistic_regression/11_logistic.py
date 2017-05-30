# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:01:02 2017

@author: tsann
original source: https://github.com/lazyprogrammer/machine_learning_examples/blob/master/ann_logistic_extra/ecommerce_data.csv
"""
import numpy as np

N = 100
D = 2

X = np.random.randn(N,D)
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis=1)

w = np.random.randn(D + 1)

z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

print(sigmoid(z))