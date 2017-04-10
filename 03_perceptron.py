# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 01:45:53 2017

@author: tsann
original source: http://www.python-course.eu/neural_networks.php
"""

import numpy as np
class Perceptron:
    
    def __init__(self, input_length, weights=None):
        if weights is None:
            self.weights = np.ones(input_length) * 0.5
        else:
            self.weights = weights
        
    @staticmethod
    def unit_step_function(x):
        if x > 0.5:
            return 1
        return 0
        
    def __call__(self, in_data):
        weighted_input = self.weights * in_data
        #print("win: " + str(weighted_input))
        #print("wis: " + str(weighted_input.sum()))
        weighted_sum = weighted_input.sum()
        #print("ws: " + str(weighted_sum))
        return Perceptron.unit_step_function(weighted_sum)
    
p = Perceptron(2, np.array([0.5, 0.5]))
data_input = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]
for x in data_input:
    y = p(np.array(x))
    print(x, y)