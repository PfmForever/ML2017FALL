import os, sys
import numpy as np
from random import shuffle
import argparse
from math import log, floor
import pandas as pd

save_dir = 'logistic_params/'

# ------ load_data
X_train = pd.read_csv(sys.argv[3], sep=',', header=0)
X_train['age_sqr'] = X_train['age'] ** 2
X_train['capital_gain_sqr'] = X_train['capital_gain'] ** 2
X_train = np.array(X_train.values)
Y_train = pd.read_csv(sys.argv[4], sep=',', header=0)
Y_train = np.array(Y_train.values)
X_test = pd.read_csv(sys.argv[5], sep=',', header=0)
X_test['age_sqr'] = X_test['age'] ** 2
X_test['capital_gain_sqr'] = X_test['capital_gain'] ** 2
X_test = np.array(X_test.values)

# --- normalize
# Feature normalization with train and test X
X_train_test = np.concatenate((X_train, X_test))
mu = (sum(X_train_test) / X_train_test.shape[0])
sigma = np.std(X_train_test, axis=0)
mu = np.tile(mu, (X_train_test.shape[0], 1))
sigma = np.tile(sigma, (X_train_test.shape[0], 1))
X_train_test_normed = (X_train_test - mu) / sigma
# Split to train, test again
X_train = X_train_test_normed[0:X_train.shape[0]]
X_test = X_train_test_normed[X_train.shape[0]:]

# train
# Split a 10%-validation set from the training set
valid_set_percentage = 0.1
# split_valid_set
all_data_size = len(X_train)
valid_data_size = int(floor(all_data_size * valid_set_percentage))

randomize = np.arange(len(X_train))
np.random.shuffle(randomize)
X_all, Y_all = (X_train[randomize], Y_train[randomize])

X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

# Initiallize parameter, hyperparameter
w = np.zeros((108,))
b = np.zeros((1,))
l_rate = 0.1
batch_size = 32
train_data_size = len(X_train)
step_num = int(floor(train_data_size / batch_size))
epoch_num = 1000
save_param_iter = 50

# Start training
total_loss = 0.0
for epoch in range(1, epoch_num):
    # Do validation and parameter saving
    if (epoch) % save_param_iter == 0:
        print('=====Saving Param at epoch %d=====' % epoch)
        print('epoch avg loss = %f' % (total_loss / (float(save_param_iter) * train_data_size)))
        total_loss = 0.0
        # valid
        valid_data_size = len(X_valid)

        z = (np.dot(X_valid, np.transpose(w)) + b)
        y = np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1-(1e-8))
        y_ = np.around(y)
        result = (np.squeeze(Y_valid) == y_)
        print('Validation acc = %f' % (float(result.sum()) / valid_data_size))

    # Random shuffle
    randomize = np.arange(len(X_train))
    np.random.shuffle(randomize)
    X_train, Y_train = (X_train[randomize], Y_train[randomize])

    # Train with batch
    for idx in range(step_num):
        X = X_train[idx*batch_size:(idx+1)*batch_size]
        Y = Y_train[idx*batch_size:(idx+1)*batch_size]

        z = np.dot(X, np.transpose(w)) + b
        y = np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1-(1e-8))

        cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y)) + np.dot((1 - np.squeeze(Y)), np.log(1 - y)))
        total_loss += cross_entropy
        lam = 0
        w_grad = np.mean(-1 * X * (np.squeeze(Y) - y).reshape((batch_size,1)), axis=0)+2*lam*w
        b_grad = np.mean(-1 * (np.squeeze(Y) - y))

        # SGD updating parameters
        w = w - l_rate * w_grad
        b = b - l_rate * b_grad



# infer
test_data_size = len(X_test)
# Load parameters

# predict
z = (np.dot(X_test, np.transpose(w)) + b)
y = np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1-(1e-8))
y_ = np.around(y)

output_path = sys.argv[6]
with open(output_path, 'w') as f:
    f.write('id,label\n')
    for i, v in  enumerate(y_):
        f.write('%d,%d\n' %(i+1, v))


