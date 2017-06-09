#!/usr/bin/env python3
"""
Image Reader
"""

# Author: Gao Tong

import math
from pathlib import Path
import pickle
from PIL import Image
import re

import numpy as np

def singleTunnel(x):
    if x.ndim == 2:
        return x
    else:
        return x.mean(2)

def compressImg(x):
    [r, c] = [32, 32]

    # comp = np.empty([math.floor(r/2), math.floor(c/2)])
    # for i in range(0, comp.shape[0]):
    #     for j in range(0, comp.shape[1]):
    #         comp[i, j] = (x[2*i, 2*j] + x[2*i, 2*j+1] + x[2*i+1, 2*j] \
    #                     + x[2*i+1, 2*j+1]) / 4;
    #
    # return comp.flatten()
    x = x[0:32,0:32]
    return x.flatten()

path = '../Nosie/'
testPath = path + 'TEST/'

# initialize I, X and Y
I = []
X = []
Y = []

# load first directory
p = Path(testPath)
for f in p.iterdir():
    if f.suffix in ['.png', '.jpg']:
        print(f.name)
        x = np.array(Image.open(f))
        y = compressImg(singleTunnel(x))
        I.append(x)
        X.append(y)
        Y.append(int(re.split(r'[\[\]]', f.name)[1]))

d = {'data':X, 'images':I, 'target':Y}

with open('../data/test_digits', mode='wb') as fout:
    pickle.dump(d, fout)
