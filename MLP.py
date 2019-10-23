#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 23:28:35 2019

@author: chenxuezhi
"""


import numpy as np
import matplotlib.pyplot as plt
from random import random
from random import seed
from math import exp


data = np.genfromtxt('perceptron1.txt',delimiter = '')
train = data[:,:2]
label = data[:,2]
train_c = np.concatenate((-(np.ones(shape = (train.shape[0],1))),train),axis = 1).astype(float)
train_set = np.array(train_c[:int(2/3*train_c.shape[0]),:])
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
    
def activation(neurons, inputs):
    sum = 0
    X = np.array(inputs)
    weight = np.array(neurons['weights'])
    sum += np.dot(weight,X.T)
    return sum

def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))

def forward_propagation(network,inputs):
    inputss = inputs
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activate = activation(neuron, inputss)
            neuron['output'] = sigmoid(activate)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

Output = []
for i in range(train_set.shape[0]):
    Output.append(forward_propagation(network,train_set[i]))
print(Output)
            
            
            
            

