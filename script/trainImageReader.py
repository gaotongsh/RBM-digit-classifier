#!/usr/bin/env python3
"""
Image Reader
"""

# Author: Gao Tong

import math
from pathlib import Path
import pickle
from PIL import Image

import numpy as np

def singleTunnel(x):
    if x.ndim == 3:
        x = x.mean(2)
    return x

def compressImg(x):
    [r, c] = [32, 32]
    # x = x[0:r, 0:c]
    # return x.flatten()

    comp = np.empty([math.floor(r/2), math.floor(c/2)])
    for i in range(0, comp.shape[0]):
        for j in range(0, comp.shape[1]):
            # comp[i, j] = (x[2*i, 2*j] + x[2*i, 2*j+1] + x[2*i+1, 2*j] \
            #             + x[2*i+1, 2*j+1]) / 4;
            comp[i, j] = x[2*i, 2*j];

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
        x = np.array(Image.open(f))
        y = compressImg(singleTunnel(x))
        I.append(x)
        X.append(y)
        Y.append(int(f.name[0]))

# load second directory
trainDigitPath = trainPath + 'hjk_picture'
p = Path(trainDigitPath)
for f in p.iterdir():
    if f.suffix in ['.png', '.jpg']:
        print(f.name)
        x = np.array(Image.open(f))
        y = compressImg(singleTunnel(x))
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
        x = np.array(Image.open(f))
        y = compressImg(singleTunnel(x))
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
        x = np.array(Image.open(f))
        y = compressImg(singleTunnel(x))
        I.append(x)
        X.append(y)
        Y.append(int(f.name[2]))

d = {'data':X, 'images':I, 'target':Y}

with open('../data/digits', mode='wb') as fout:
    pickle.dump(d, fout)
