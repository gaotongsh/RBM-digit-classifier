#!/usr/bin/env python3
"""
Image Reader
"""

# Author: Gao Tong

import math
from pathlib import Path
import pickle
import re

import matplotlib.image as mpimg
import numpy as np

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
        with f.open(mode='rb') as fin:
            # try:
            x = singleTunnel(mpimg.imread(fin))
            # except Exception as e:
            #     pass
        y = compressImg(x)
        I.append(x)
        X.append(y)
        Y.append(re.split(r'[\[\]]', f.name)[1])

d = {'data':X, 'images':I, 'target':Y}

with open('../data/test_digits', mode='wb') as fout:
    pickle.dump(d, fout)
