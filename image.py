# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 13:24:59 2020

@author: JJR
"""

import os
import numpy as np
from PIL import Image
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU, LSTM
from keras.layers import SimpleRNN

def get_imlist(path):
    
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.bmp')]

def readfile(filetype):
    n = 0
    path = (r'C:/Users/JJR/Desktop/image_all/')
    train_image = path + filetype
    file_list = []
    for f in os.listdir(train_image):
        n = n + 1
        train_image_path = path + filetype + '/' + str(f)
        for number in range(10):
            number_image_path = train_image_path + '/' + str(number)
            file_list.append(number_image_path)
    data = np.empty((n*5*10,28*28))
    data_label = np.empty((n*5*10,1))
    d = 0
    a = 0
    while (d < n*5*10):       
        for i in file_list:    
            if file_list.index(i) % 10 == a:
                c = get_imlist(i)
                for i in range(len(c)):
                    img = Image.open(c[i])
                    img_ndarray=np.asarray(img,dtype='float64')/255
                    data[d] = np.ndarray.flatten(img_ndarray)
                    data_label[d] = a
                    d = d + 1
        a = a + 1
    
    return data,data_label
        
    
train_feature,train_label = readfile('image')  
test_feature,test_label = readfile('test_image')  

train_feature_reshape = train_feature.reshape(-1, 28, 28)
test_feature_reshape = test_feature.reshape(-1, 28, 28)

train_label_onehot = np_utils.to_categorical(train_label)
test_label_onehot = np_utils.to_categorical(test_label)

model = Sequential()
model.add(GRU(units=256, input_shape=(28, 28), unroll = True))
model.add(Dropout(0.1))
model.add(Dense(units=10, activation='softmax', kernel_initializer = 'normal'))
model.summary()


model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

train_history = model.fit(x = train_feature_reshape, 
                          y = train_label_onehot,
                          validation_split = 0.2,
                          epochs = 10,
                          batch_size = 200,
                          verbose = 2)
            
scores = model.evaluate(test_feature_reshape,test_label_onehot)
print(scores[1])            
    
    
        
            
        
    
    
     
    