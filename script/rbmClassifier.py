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
import scipy.io as sio

with open('../data/digits', mode='rb') as fin:
    digits = pickle.load(fin)

X_train = np.asarray(digits['data'], 'float32')
Y_train = digits['target']

# train = sio.loadmat('../data/train.mat')
# X_train = train['data']
X_train = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) + 0.0001)
# Y_train = train['label']

rbm = BernoulliRBM(random_state=0, verbose=True)
logistic = linear_model.LogisticRegression()
classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

# Training

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = 0.04
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 3000
logistic.C = 5000.0

# Training RBM-Logistic Pipeline
print("Training")
classifier.fit(X_train, Y_train)

with open('../data/test_digits', mode='rb') as fin:
    test_digits = pickle.load(fin)

X_test = test_digits['data']
# np.asarray(test_digits['data'], 'float32')
Y_test = test_digits['target']

# test = sio.loadmat('../data/test.mat')
# X_test = test['test']
X_test = (X_test - np.min(X_test, 0)) / (np.max(X_test, 0) + 0.0001)
# Y_test = test['test_label']

print()
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        classifier.predict(X_test))))
