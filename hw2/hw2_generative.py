import os, sys
import pandas as pd
import numpy as np
from random import shuffle
import argparse
from math import log, floor

save_dir = 'generative_params/'
output_dir = 'generative_output/'

# -------------------------load_data-------------------------
X_train = pd.read_csv(sys.argv[3], sep=',', header=0)
X_train = np.array(X_train.values)
Y_train = pd.read_csv(sys.argv[4], sep=',', header=0)
Y_train = np.array(Y_train.values)
X_test = pd.read_csv(sys.argv[5], sep=',', header=0)
X_test = np.array(X_test.values)

# -------------------------normalize-------------------------
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

# -------------------------train-------------------------
# Split a 10%-validation set from the training set

X_train, Y_train, X_valid, Y_valid = [X_train, Y_train,X_train, Y_train] 

# Gaussian distribution parameters
train_data_size = X_train.shape[0]
cnt1 = 0
cnt2 = 0

mu1 = np.zeros((106,))
mu2 = np.zeros((106,))
for i in range(train_data_size):
    if Y_train[i] == 1:
        mu1 += X_train[i]
        cnt1 += 1
    else:
        mu2 += X_train[i]
        cnt2 += 1
mu1 /= cnt1
mu2 /= cnt2

sigma1 = np.zeros((106,106))
sigma2 = np.zeros((106,106))
for i in range(train_data_size):
    if Y_train[i] == 1:
        sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [(X_train[i] - mu1)])
    else:
        sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [(X_train[i] - mu2)])
sigma1 /= cnt1
sigma2 /= cnt2
shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2
N1 = cnt1
N2 = cnt2

print('=====Validating=====')
sigma_inverse = np.linalg.inv(shared_sigma)
w = np.dot( (mu1-mu2), sigma_inverse)
x = X_valid.T
b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
a = np.dot(w, x) + b
y = np.clip(1 / (1.0 + np.exp(-a)), 1e-8, 1-(1e-8))
y_ = np.around(y)
result = (np.squeeze(Y_valid) == y_)
print('Valid acc = %f' % (float(result.sum()) / result.shape[0]))



# Predict
sigma_inverse = np.linalg.inv(shared_sigma)
w = np.dot( (mu1-mu2), sigma_inverse)
x = X_test.T
b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
a = np.dot(w, x) + b
y = np.clip(1 / (1.0 + np.exp(-a)), 1e-8, 1-(1e-8))
y_ = np.around(y)

print('=====Write output to %s =====' % output_dir)
# Write output
f = open(sys.argv[6], 'w')
f.write('id,label\n')
for i, v in  enumerate(y_):
    f.write('%d,%d\n' %(i+1, v))