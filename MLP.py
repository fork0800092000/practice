# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 20:05:20 2019

@author: aaa
"""

import numpy as np
import matplotlib.pyplot as plt
from random import random
from random import seed


data = np.genfromtxt('perceptron1.txt',delimiter = '')
train = data[:,:2]
label = data[:,2]
train_c = np.concatenate((-(np.ones(shape = (train.shape[0],1))),train),axis = 1).astype(float)
train_set = train_c[:int(2/3*train_c.shape[0]),:]
train_label = label[:int(2/3*label.shape[0]),]
test_set = train_c[int(2/3*train_c.shape[0]):,:]

#Initialize Network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

seed(1)
network = initialize_network(2,2,1)
for layer in network:
    print(layer)
    
def activation(weights, inputs):
    sum = 0
    sum += np.dot(weights,inputs.T)
    return sum
def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))