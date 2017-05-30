"""
run neural network
~~~~~~~~~~~~


original source: http://neuralnetworksanddeeplearning.com
"""

# load data
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network

print("epoch 30, mini batch 10 learning rate 3.0")
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

print("epoch 100, mini batch 10 learning rate 3.0")
net = network.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

print("epoch 100, mini batch 10 learning rate 0.001")
net = network.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 0.001, test_data=test_data)

print("epoch 30, mini batch 10 learning rate 100.0")
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 100.0, test_data=test_data)