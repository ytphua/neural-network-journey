# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 01:45:53 2017

@author: tsann
original source: http://www.python-course.eu/neural_networks.php
"""

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
class Perceptron:

    def __init__(self, input_length, weights=None):
        if weights==None:
            self.weights = np.random.random((input_length)) * 2 - 1
        self.learning_rate = 0.1
        
    @staticmethod
    def unit_step_function(x):
        if x < 0:
            return 0
        return 1
        
    def __call__(self, in_data):
        weighted_input = self.weights * in_data
        weighted_sum = weighted_input.sum()
        return Perceptron.unit_step_function(weighted_sum)
    
    def adjust(self, 
               target_result, 
               calculated_result,
               in_data):
        error = target_result - calculated_result
        for i in range(len(in_data)):
            correction = error * in_data[i] *self.learning_rate
            self.weights[i] += correction 

     
def above_line(point, line_func):
    x, y = point
    if y > line_func(x):
        return 1
    else:
        return 0

from matplotlib import pyplot as plt
class1 = [(3, 4), (4.2, 5.3), (4, 3), (6, 5), (4, 6), (3.7, 5.8),
          (3.2, 4.6), (5.2, 5.9), (5, 4), (7, 4), (3, 7), (4.3, 4.3) ] 
class2 = [(-3, -4), (-2, -3.5), (-1, -6), (-3, -4.3), (-4, -5.6), 
          (-3.2, -4.8), (-2.3, -4.3), (-2.7, -2.6), (-1.5, -3.6), 
          (-3.6, -5.6), (-4.5, -4.6), (-3.7, -5.8) ]
X, Y = zip(*class1)
plt.scatter(X, Y, c="r")
X, Y = zip(*class2)
plt.scatter(X, Y, c="b")
plt.show()

from itertools import chain
p = Perceptron(2)
def lin1(x):
    return  x + 4
for point in class1:
    p.adjust(1, 
             p(point), 
             point)
for point in class2:
    p.adjust(0, 
             p(point), 
             point)
    
evaluation = Counter()
for point in chain(class1, class2):
    if p(point) == 1:
        evaluation["correct"] += 1
    else:
        evaluation["wrong"] += 1
        
testpoints = [(3.9, 6.9), (-2.9, -5.9)]
for point in testpoints:
    print(p(point))
        
print(evaluation.most_common())