# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:01:02 2017

@author: tsann
original source: https://github.com/lazyprogrammer/machine_learning_examples/blob/master/ann_logistic_extra/ecommerce_data.csv
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
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
X, Y =  shuffle(X, Y)

Xtrain = X[:-100]
Ytrain = Y[:-100]
Xtest = X[-100]
Ytest = Y[-100]

D= X.shape[1]
W = np.random.randn(D)
b = 0

# make predictions
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

# calculate the accuracy
def classification_rate(Y, P):
    return np.mean(Y == P)

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY) + (1-T)*np.log(1-pY))

train_costs = []
test_costs = []
learning_rate = 0.001
for i in range(10000):
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)
    
    ctrain = cross_entropy(Ytrain, pYtrain)
    ctest = cross_entropy(Ytest, pYtest)
    
    train_costs.append(ctrain)
    test_costs.append(ctest)
    
    W -= learning_rate*Xtrain.T.dot(pYtrain - Ytrain)
    b -= learning_rate*(pYtrain - Ytrain).sum()
    
    if i % 1000 == 0:
        print(i, ctrain, ctest)

print("Final train classification rate: ", classification_rate(Ytrain, np.round(pYtrain)))
print("Final test classification rate: ", classification_rate(Ytest, np.round(pYtest)))

legend1, = plt.plot(train_costs, label="train cost")
legend2, = plt.plot(test_costs, label="test cost")
plt.legend([legend1, legend2])