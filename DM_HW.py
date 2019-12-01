# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 13:34:11 2019

@author: aaa
"""
import numpy as np

'''data = np.loadtxt('t2.txt', delimiter = ",",usecols = (0,))'''
simDat = []
with open('t2.txt') as f:
    for line in f.read().splitlines():
        simDat.append(line[5:].split(','))

def createInitSet(dataset):
    retDict = {}
    for trans in dataset:
        retDict[frozenset(trans)] = 1
    return retDict

initset = createInitSet(simDat)