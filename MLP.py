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


'''
traindata = data[:,:2]
train_max = np.max(traindata, axis = 0)
train_min = np.min(traindata, axis = 0)
for j in range(traindata.shape[1]):
    for i in range(len(traindata)):
        traindata[i][j] = (traindata[i][j] - train_min[j]) / (train_max[j] - train_min[j])
   
label = data[:,2]

for i in range(len(label)):
    if label[i] == 1:
        label[i] = 0
    else:
        label[i] = 1

label_max = np.max(label, axis = 0)
label_min = np.min(label, axis = 0)
for i in range(len(label)):
        label[i] = (label[i] - label_min) / (label_max - label_min)



train_c = np.concatenate((-(np.ones(shape = (train.shape[0],1))),train),axis = 1).astype(float)
train_set = train_c[:int(2/3*train_c.shape[0]),:]
train_label = label[:int(2/3*label.shape[0]),]
test_set = train_c[int(2/3*train_c.shape[0]):,:]
'''
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[int(row[-1])] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

seed(1)
n_inputs = len(data[0]) - 1
n_outputs = len(set(row[-1] for row in data))
network = initialize_network(n_inputs,2,n_outputs)
for layer in network:
    print(layer)
print('\n',data)
train_network(network, data, 0.4, 20, n_outputs)
for layer in network:
    print(layer)
  
    
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores   

def predict(network, row):
    output = forward_propagate(network, row)
    return output.index(max(output))   

def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

for row in data:
    prediction = predict(network, row)
    print('Expected = %d, Got = %d' % (row[-1], prediction))
    



        
        
        
        
        
        
        
        
        
        
        
        
        