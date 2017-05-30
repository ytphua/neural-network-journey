# -*- coding: utf-8 -*-
"""
Created on Mon May  8 04:23:11 2017

@author: tsann
original source: https://github.com/lazyprogrammer/machine_learning_examples
"""
# forward propagation example for deep learning in python class.
#
# the notes for this class can be found at: 
# https://deeplearningcourses.com/c/data-science-deep-learning-in-python
# https://www.udemy.com/data-science-deep-learning-in-python

import numpy as np
import pandas as pd

df = pd.read_csv('ecommerce_data.csv')
print(df.head())

def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    
    # just in case you're curious what's in it
    # df.head()

    # easier to work with numpy array
    data = df.as_matrix()

    X = data[:,:-1]
    Y = data[:,-1]

    # normalize columns 1 and 2
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()

    # create a new matrix X2 with the correct number of columns
    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:,0:(D-1)] = X[:,0:(D-1)] # non-categorical

    # one-hot
    for n in range(N):
        t = int(X[n,D-1])
        X2[n,t+D-1] = 1

    # method 2
    # Z = np.zeros((N, 4))
    # Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
    # # assign: X2[:,-4:] = Z
    # assert(np.abs(X2[:,-4:] - Z).sum() < 1e-10)

    return X2, Y

def get_binary_data():
    # return only the data from the first 2 classes
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2

X, Y = get_data()

M = 5
#import matplotlib.pyplot as plt
#
#Nclass = 500
#
#X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
#X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
#X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
#X = np.vstack([X1, X2, X3])
#
#Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
#
## let's see what it looks like
#plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
#plt.show()
#
# randomly initialize weights
D = X.shape[1] # dimensionality of input
M = 5 # hidden layer size
K = len(set(Y))# number of classes
W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)

def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2)+b2)

# determine the classification rate
# num correct / num total
def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total

P_Y_given_X = forward(X, W1, b1, W2, b2)
Prediction = np.argmax(P_Y_given_X, axis=1)

# verify we chose the correct axis
assert(len(Prediction) == len(Y))

print("Classification rate for randomly chosen weights:", classification_rate(Y, P))