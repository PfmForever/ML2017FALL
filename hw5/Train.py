import pandas as pd
import numpy as np
from pandas import Series, DataFrame 
import pickle
from keras.utils import np_utils
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Flatten, Dot, Add, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as BK
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
def rmse(y_true, y_pred):
    return BK.sqrt(BK.mean((y_pred-y_true) ** 2))
model_dir = './dim2000_model.{epoch:02d}-{val_rmse:.4f}.h5'
dimension = 2000
with open('x.pkl','rb') as handle:
    x = pickle.load(handle)
with open('y.pkl','rb') as handle:
    y = pickle.load(handle)

randomIdx = np.arange(len(x))
np.random.shuffle(randomIdx)
x = x[randomIdx]
y = y[randomIdx]

user_input = Input(shape = (1,))
movie_input = Input(shape = (1,))
user_bias= Input(shape = (1,))
movie_bias= Input(shape = (1,))
user_v = Embedding(input_dim = 6041,output_dim = dimension,embeddings_regularizer = l2(0.000005),input_length = 1)(user_input)
user_v = Flatten()(user_v)
movie_v = Embedding(input_dim = 3953,output_dim = dimension,embeddings_regularizer = l2(0.000005),input_length = 1)(movie_input)
user_bias = Embedding(input_dim = 6041,output_dim = 1,embeddings_initializer = 'zero')(user_bias)
user_bias = Flatten()(user_bias)
movie_bias = Embedding(input_dim = 3953,output_dim = 1,embeddings_initializer = 'zero')(movie_bias)
movie_bias = Flatten()(movie_bias)
movie_v = Flatten()(movie_v)


dot = Dot(axes = 1)([user_v, movie_v])
add = Add()([dot,user_bias,movie_bias])
model = Model([user_input, movie_input], dot)
model.summary()
opt = Adam(lr = 0.0005)
model.compile(optimizer = opt, loss = 'mse', metrics = [rmse])
ES = EarlyStopping(monitor = 'val_rmse',
        patience = 5,
        verbose = 1)
cp = ModelCheckpoint(model_dir,
        monitor = 'val_rmse',
        save_best_only = True,
        verbose = 0)
reduce_lr = ReduceLROnPlateau(factor = 0.5, patience = 3)
his = model.fit(np.hsplit(x, 2), y,
        batch_size = 1024, 
        epochs = 1000, 
        validation_split = 0.1,
        callbacks = [ES, cp, reduce_lr])
