import numpy as np
x = np.load('./facial_data_X.npy')
y = np.load('./facial_labels.npy')
# x -= np.mean(x, axis=0)
# x /= np.std(x, axis=0)
x = x/255
def ExportCSV(array,name):
    df = pandas.DataFrame(array)
    df.to_csv(name)

import os
from keras.layers import Dense, Convolution2D, UpSampling2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout, Reshape
from keras.models import Sequential
from keras.utils import np_utils










X_train = x[0:28610,:]
Y_train = y[0:28610]
print(X_train.shape , Y_train.shape)
X_crossval = x[28610:28710,:]
Y_crossval = y[28610:28710]
# print (X_crossval.shape , Y_crossval.shape)
X_train = X_train.reshape((X_train.shape[0], 48, 48, 1))
X_crossval = X_crossval.reshape((X_crossval.shape[0], 48, 48,1))
import tensorflow as tf


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.regularizers import l2
import numpy
import csv
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
img_rows, img_cols = 48, 48
model = Sequential()
model.add(Convolution2D(64, (5, 5),input_shape=(img_rows, img_cols,1),padding = 'valid'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(Convolution2D(64, 3, 3))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf')) 
model.add(Convolution2D(64, 3, 3))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(Convolution2D(128, 3, 3))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(Convolution2D(128, 3, 3))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(Dropout(0.2))
model.add(Dense(1024))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(Dropout(0.2))


model.add(Dense(7))


model.add(Activation('softmax'))

ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print(y.shape)
y_ = np_utils.to_categorical(y, num_classes=7)
print(y_.shape)
Y_train = y_[:28610]
Y_crossval = y_[28610:28710]
# print(X_crossval.shape, model.input_shape, Y_crossval.shape)
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
ModelCheckPoint_filename = 'm_model.{epoch:02d}-{val_acc:.4f}-adam.h5' 
checkpointer = [
TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True),
ModelCheckpoint(ModelCheckPoint_filename, monitor='val_loss', verbose=1, save_best_only=False, mode='auto', period=1),
ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50, verbose=1)]
model.fit_generator(datagen.flow(X_train, Y_train,
                    batch_size=128),
                    validation_data=(X_crossval, Y_crossval),
                    nb_epoch=280,
                    samples_per_epoch=X_train.shape[0],
                    callbacks=checkpointer)
model.save("m_model.279-0.7172-adam.h5")

