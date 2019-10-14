# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 11:17:34 2019

@author: aaa
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('perceptron1.txt', delimiter = " ", dtype = float)

train_data = data[:, :2]
label_data = data[:, 2]

class perceptron(object):
    def __init__(self, w, b = 1, lr = 1000, epoch = 1):
        self.w = np.array([0,0])
        self.b = b
        self.lr = lr
        self.epoch = epoch
    def predict(self,X):
        summation = np.dot(self.w.T,X) + self.b
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation
    def fit(self, training_inputs, labels):
        for i in range(self.epoch):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                if (label != prediction)&(label==1) :
                    self.w = self.w + self.lr * inputs 
                    self.b = self.b - self.lr * 1
                elif (label != prediction)&(label==0):
                    self.w = self.w - self.lr * inputs
                    self.b = self.b + self.lr * 1
                else:
                    self.w = self.w
        return self.w        
p = perceptron(1)
p.fit(train_data,label_data)
x_list = train_data[:,0]
y_list = train_data[:,1]
plt.figure('Scatter fig')
ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('y')
x = np.linspace(-1,1,10)
y = (-p.b-p.w[0]*x)/p.w[1]
ax.plot(x,y)
'''def funtion(x1,x2):
    return p.w[0]*x1 + p.w[1]*x2 + 1
x = np.arange(-1,1,1)
y = np.arange(-1,1,1)
Y = funtion(x,y)
plt.plot(Y)'''
for tdata, ldata in zip(train_data, label_data):
    if ldata == 1:
        ax.scatter(tdata[0],tdata[1],c = 'r',s = 30)
    else:
        ax.scatter(tdata[0],tdata[1],c = 'b',s = 30)
plt.show()