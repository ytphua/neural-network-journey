# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 02:22:14 2017
original source: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/linear_regression_class
@author: tsann
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#loadd data
X = []
Y = []

df = pd.read_excel('mlr02.xls')
X = df.as_matrix()

plt.scatter(X[:,1], X[:,0])
plt.show()

plt.scatter(X[:,2], X[:,0])
plt.show()

df['ones'] = 1
Y = df['X1']
X = df[['X2','X3','ones']]
X2only = df[['X2','ones']]
X3only = df[['X3','ones']]

def get_r2(X,Y):
    #calculate weight
    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T,Y))
    Yhat = np.dot(X, w)
    print("w: " + str(w))
    
    ##computer r-square
    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1)/d2.dot(d2)
    
    return r2

print("R2 for x2= "+str(get_r2(X2only, Y)))
print("R2 for x3= "+str(get_r2(X3only, Y)))
print("R2 for both= "+str(get_r2(X, Y)))

