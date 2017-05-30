# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:01:02 2017

@author: tsann
original source: https://github.com/lazyprogrammer/machine_learning_examples/blob/master/ann_logistic_extra/ecommerce_data.csv
"""
import numpy as np
import pandas as pd


def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    data = df.as_matrix()
    
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

X, Y = get_binary_data()

# randomly initialize weights
D = X.shape[1]
W = np.random.randn(D)
b = 0 # bias term

# make predictions
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

P_Y_given_X = forward(X, W, b)
predictions = np.round(P_Y_given_X)

# calculate the accuracy
def classification_rate(Y, P):
    return np.mean(Y == P)

print("Score:", classification_rate(Y, predictions))
