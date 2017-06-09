#!/usr/bin/env python3
"""
Image Reader
"""

# Author: Gao Tong

import math
from pathlib import Path
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

def singleTunnel(x):
    if x.ndim == 3:
        s = x.shape[2]
        x = x.sum(2) / s
    return x

def compressImg(x):
    [r, c] = x.shape

    comp = np.empty([math.floor(r/2), math.floor(c/2)])
    for i in range(0, comp.shape[0]):
        for j in range(0, comp.shape[1]):
            comp[i, j] = round((x[i, j] + x[i, j+1] + x[i+1, j] + x[i+1, j+1]) / 4);

    return comp.flatten()

path = '../Nosie/'
trainPath = path + 'TRAIN/'

# initialize I, X and Y
I = []
X = []
Y = []

# load first directory
trainDigitPath = trainPath + 'digits'
p = Path(trainDigitPath)
for f in p.iterdir():
    if f.suffix in ['.png', '.jpg']:
        print(f.name)
        with f.open(mode='rb') as fin:
            x = singleTunnel(mpimg.imread(fin))
        y = compressImg(x)
        I.append(x)
        X.append(y)
        Y.append(int(f.name[0]))

# load second directory
trainDigitPath = trainPath + 'hjk_picture'
p = Path(trainDigitPath)
for f in p.iterdir():
    if f.suffix in ['.png', '.jpg']:
        print(f.name)
        with f.open(mode='rb') as fin:
            x = singleTunnel(mpimg.imread(fin))
        y = compressImg(x)
        I.append(x)
        X.append(y)
        if f.name[1] == '_':
            Y.append(int(f.name[0]))
        else:
            Y.append(int(f.name[2]))

# load third directory
trainDigitPath = trainPath + 'Li Wanjin'
p = Path(trainDigitPath)
for f in p.iterdir():
    if f.suffix in ['.png', '.jpg']:
        print(f.name)
        with f.open(mode='rb') as fin:
            x = singleTunnel(mpimg.imread(fin))
        y = compressImg(x)
        I.append(x)
        X.append(y)
        if f.name[1] == '-':
            Y.append(int(f.name[2]))
        else:
            Y.append(int(f.name[0]))

# load forth directory
trainDigitPath = trainPath + 'number'
p = Path(trainDigitPath)
for f in p.iterdir():
    if f.suffix in ['.png', '.jpg']:
        print(f.name)
        with f.open(mode='rb') as fin:
            x = singleTunnel(mpimg.imread(fin))
        y = compressImg(x)
        I.append(x)
        X.append(y)
        Y.append(int(f.name[2]))

d = {'data':X, 'images':I, 'target':Y}

with open('../data/digits', mode='wb') as fout:
    pickle.dump(d,fout)
