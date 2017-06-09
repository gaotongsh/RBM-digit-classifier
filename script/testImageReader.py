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
    if x.ndim == 3:
        x = x.mean(2)
    return x

def compressImg(x):
    [r, c] = [32, 32]
    const = 3

    comp = np.empty([math.floor(r/const), math.floor(c/const)])
    for i in range(0, comp.shape[0]):
        for j in range(0, comp.shape[1]):
            comp[i, j] = x[const*i, const*j];

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
        x = np.array(Image.open(f))
        y = compressImg(singleTunnel(x))
        I.append(x)
        X.append(y)
        Y.append(int(re.split(r'[\[\]]', f.name)[1]))

d = {'data':X, 'images':I, 'target':Y}

with open('../data/test_digits', mode='wb') as fout:
    pickle.dump(d, fout)
