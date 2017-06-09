#!/usr/bin/env python3
"""
Core RBM classifier
"""

# Author: Gao Tong

import pickle

import numpy as np
from sklearn import linear_model, metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

with open('../data/digits', mode='rb') as fin:
    digits = pickle.load(fin)

X_train = digits['data']
Y_train = digits['target']
X_train = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) + 0.0001)

rbm = BernoulliRBM(random_state=0, verbose=True)
logistic = linear_model.LogisticRegression()
classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

# Training

# Hyper-parameters.
rbm.learning_rate = 0.04
rbm.n_iter = 20
rbm.n_components = 10000
logistic.C = 5000.0

# Training RBM-Logistic Pipeline
print("Training")
classifier.fit(X_train, Y_train)

with open('../data/test_digits', mode='rb') as fin:
    test_digits = pickle.load(fin)

X_test = test_digits['data']
Y_test = test_digits['target']
X_test = (X_test - np.min(X_test, 0)) / (np.max(X_test, 0) + 0.0001)

print()
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        classifier.predict(X_test))))
