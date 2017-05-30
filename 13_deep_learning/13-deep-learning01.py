# -*- coding: utf-8 -*-
"""
Created on Mon May  8 04:23:11 2017

@author: tsann
original source: https://github.com/lazyprogrammer/machine_learning_examples
"""

import numpy as np

a = np.random.randn(100,5)
print(a)

expa = np.exp(a)
print(expa)

answer = expa/expa.sum(axis=1, keepdims=True)
print(answer)
print(answer.sum())
print(answer.sum(axis=1))
print(answer.sum(axis=1, keepdims=True))
print(answer.sum(axis=1, keepdims=True).shape)