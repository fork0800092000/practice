#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 23:28:35 2019

@author: chenxuezhi
"""



import numpy as np
import matplotlib.pyplot as plt
from random import random,sample
from random import seed
from math import exp
import random

def normalize_data(data_norm):
    data_max = np.max(data_norm, axis = 0)
    data_min = np.min(data_norm, axis = 0)
    datac = np.empty(shape = (data_norm.shape[0],data_norm.shape[1]), dtype = float)
    for j in range(data_norm.shape[1] - 1):
        for i in range(len(data_norm)):
            data_norm[i][j] = (data_norm[i][j] - data_min[j]) / (data_max[j] - data_min[j])
    return data_norm

def split(data):
    r = random.sample(range(len(data)), int(2/3 * data.shape[0]))
    s = random.sample(range(len(data)), int(1/3 * data.shape[0]))
    train_data = data[:int(2/3 * data.shape[0]),:]
    test_data = data[int(1/3 * data.shape[0]):,:]
    return train_data, test_data


def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	hidden_layer = [{'weights':[random.random() for i in range(n_hidden + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	hidden_layer = [{'weights':[random.random() for i in range(n_hidden + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random.random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
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
	'print(inputs)'
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
		''''print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))'''

def evaluate_algorithm(dataset, algorithm, *args):
    train_set, test_set = split(dataset)
    predicted, network = algorithm(dataset, train_set, test_set, *args)
    actual = [row[-1] for row in test_set]
    'print(list(zip(actual, predicted)))'
    accuracy = accuracy_metric(actual, predicted)
    scores = list()
    scores.append(accuracy)
    return scores, predicted, network

def back_propagation(datset, train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in datset]))
	print(n_outputs)
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	print(network)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		'print(prediction,end = '')'
		predictions.append(prediction)
	return predictions, network

def predict(network, row):
    output = forward_propagate(network, row)
    print (output)
    return output.index(max(output))

def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

def change_label(dataset):
    category = range(len(set(row[-1] for row in dataset)))
    label = list()
    for row in dataset:
        if row[-1] not in label:
            label.append(row[-1])
    for i in range(len(label)):
        for row in dataset:
            if row[-1] == label[i]:
                row[-1] = category[i]

seed(2)
data = np.genfromtxt('2Circle2.txt',delimiter = '')
change_label(data)
normalize_data(data)
l_rate = 0.5
n_epoch = 100
n_hidden = 5
scores, predicted, network = evaluate_algorithm(data, back_propagation, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuraccy: %.3f%%\n'%(sum(scores)/float(len(scores))))
for i in range(len(network)):
    print('layer%d'%i)
    for j in range(len(network[i])):
        print(network[i][j]['weights'])
'print(predicted)'


data_un = np.genfromtxt('2Circle2.txt',delimiter = '')
change_label(data_un)
train, test = split(data_un)
for i in range(len(test)):
    if predicted[i] == 0:
        X = test[i][0]
        Y = test[i][1]
        plt.scatter(X, Y, c = 'blue')
    elif predicted[i] == 1:
        X = test[i][0]
        Y = test[i][1]
        plt.scatter(X, Y, c = 'red')
    else:
        X = test[i][0]
        Y = test[i][1]
        plt.scatter(X, Y, c = 'green')
plt.show()
            
            
            
            

