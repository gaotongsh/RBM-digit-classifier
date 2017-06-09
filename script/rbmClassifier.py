#!/usr/bin/env python3
"""
Core RBM classifier
"""

# Author: Gao Tong

import pickle

with open('../data/digits', mode='rb') as fin:
    digits = pickle.load(fin)
