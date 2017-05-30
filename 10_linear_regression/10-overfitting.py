# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 02:22:14 2017
original source: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/linear_regression_class
@author: tsann
"""

import numpy as np
import matplotlib.pyplot as plt


def make_poly(X, deg):
    n = len(X)
    data = [np.ones(n)]
    for d in range(deg):
        data.append(X**(d+1))
    return np.vstack(data).T

def fit(X, Y):
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T,Y))

def fit_and_display(X, Y, sample, deg):
    N = len(X)
    train_idx = np.random.choice(N, sample)
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]
    
    plt.scatter(Xtrain, Ytrain)
    plt.show()
    
    #fit polynomial
    Xtrain_poly = make_poly(Xtrain, deg)
    w = fit(Xtrain_poly, Ytrain)
    
    #display the polynomial
    X_poly = make_poly(X, deg)
    Y_hat = X_poly.dot(w)
    plt.plot(X, Y)
    plt.plot(X, Y_hat)
    plt.scatter(Xtrain, Ytrain)
    plt.title("deg = %d" % deg)
    plt.show()
    
def get_mse(Y, Yhat):
    d = Y - Yhat
    return d.dot(d) / len(d)

def plot_train_vs_test_curves(X, Y, sample=20, max_deg=20):
    N =len(X)
    train_idx = np.random.choice(N, sample)
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]
    
    test_idx = [idx for idx in range(N) if idx not in train_idx]
    
    Xtest = X[test_idx]
    Ytest = Y[test_idx]
    
    mse_trains = []
    mse_tests = []
    
    for deg in range(max_deg+1):
        Xtrain_poly = make_poly(Xtrain, deg)
        w = fit(Xtrain_poly, Ytrain)
        Yhat_train = Xtrain_poly.dot(w)
        mse_train = get_mse(Ytrain, Yhat_train)

        Xtest_poly = make_poly(Xtest, deg)
        Yhat_test = Xtest_poly.dot(w)
        mse_test = get_mse(Ytest, Yhat_test)
        
        mse_trains.append(mse_train)
        mse_tests.append(mse_test)

    plt.plot(mse_trains, label="train mse")
    plt.plot(mse_tests, label="test mse")
    plt.legend()
    plt.show()

    plt.plot(mse_trains, label="train mse")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # make up some data and plot it
    N = 100
    X = np.linspace(0, 6*np.pi, N)
    Y = np.sin(X)
    
    plt.plot(X, Y)
    plt.show()
    
    for deg in (5, 6, 7, 8, 9):
    		fit_and_display(X, Y, 10, deg)
    plot_train_vs_test_curves(X, Y)

#
##loadd data
#X = []
#Y = []
#
#df = pd.read_excel('mlr02.xls')
#X = df.as_matrix()
#
#plt.scatter(X[:,1], X[:,0])
#plt.show()
#
#plt.scatter(X[:,2], X[:,0])
#plt.show()
#
#df['ones'] = 1
#Y = df['X1']
#X = df[['X2','X3','ones']]
#X2only = df[['X2','ones']]
#X3only = df[['X3','ones']]
#
#def get_r2(X,Y):
#    #calculate weight
#    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T,Y))
#    Yhat = np.dot(X, w)
#    print("w: " + str(w))
#    
#    ##computer r-square
#    d1 = Y - Yhat
#    d2 = Y - Y.mean()
#    r2 = 1 - d1.dot(d1)/d2.dot(d2)
#    
#    return r2
#
#print("R2 for x2= "+str(get_r2(X2only, Y)))
#print("R2 for x3= "+str(get_r2(X3only, Y)))
#print("R2 for both= "+str(get_r2(X, Y)))

