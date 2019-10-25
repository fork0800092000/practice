# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 20:05:20 2019

@author: aaa
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

   
def activation(weights, inputss):
    mul = 0
    for i in range(len(weights)):
        mul += weights[i] * inputss[i]
    return mul

def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))

def forward_propagation(network,row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activate = activation(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activate)
            new_inputs.append(neuron['output'])
        new_inputs.insert(0, (-1))
        inputs = new_inputs
    return inputs[1]

def sigmoid_derivative(output):
    return output * (1.0 - output)

def backward_propagation(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != (len(network)-1):
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])
       
   
def update_weight(network, l_rate, inputs):
    for i in range(len(network)):
        bias = inputs[0:]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(1,len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][0] += l_rate * neuron['delta']
        
def train_network(network, epoch, l_rate, train, label):
    for ep in range(epoch):
        sum_error = 0
        for i in range(len(train)):
            row = train[i]
            expected = label[i]
            realout = forward_propagation(network,row)
            sum_error += sum([(expected-realout)**2])            
            backward_propagation(network,expected)
            update_weight(network, l_rate, row)
        print('>epoch = %d, l_rate = %.3f, error = %.3f\n'%(ep, l_rate, sum_error))

train_network(network, 20, 0.5, train_set, train_label)
for layer in network:
    print(layer)    

def predict(network, row):
    output = forward_propagation(network, row)
    return (output)   
print('\n')
for row in train_set:
    i = 0 
    prediction = predict(network, row)
    print('Expected = %d, Got = %d'%(train_label[i], prediction))
    i += 1

def accuracy(predict, actual):
    correct = 0
    for i in range(len(preidct)):
        if predict[i] == actual[i]:
            correct += 1
    return correct/float(len(predict)) * 100.0


        
        
        
        
        
        
        
        
        
        
        
        
        