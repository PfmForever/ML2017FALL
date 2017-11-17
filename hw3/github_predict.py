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
import pandas
from keras.preprocessing.image import ImageDataGenerator
import sys
import csv
def ExportCSV(array,name):
    df = pandas.DataFrame(array)
    df.to_csv(name)

def ExportFinalCSV(array,name):
    df = pandas.DataFrame(array)
    df.to_csv(name)

x_test = pandas.read_csv(sys.argv[1])
test_data = x_test.values
print(test_data.shape)
test_pixels = test_data[:, 1]
test_X = np.zeros((test_pixels.shape[0], 48*48))

for ix in range(test_X.shape[0]):
    p = test_pixels[ix].split(' ')
    for iy in range(test_X.shape[1]):
        test_X[ix, iy] = int(p[iy])

test_x = test_X
test_x = test_x/255
X_test = test_x.reshape((test_x.shape[0], 48, 48, 1))
print(X_test[0])

model = keras.models.load_model("./m_model.279-0.7172-adam.h5")

prediction = model.predict(X_test)
ExportCSV(prediction,sys.argv[2]+'-temp')

res = pandas.read_csv(sys.argv[2]+'-temp')
res = res.values
print(res.shape)
count = np.zeros([7178,2])
print(count.shape)
for idx in range(res.shape[0]):
    count[idx,0] = idx    
    count[idx,1] = np.argmax(res[idx,1:8])

count = count.astype(np.int64)
filename = sys.argv[2]
text = open(filename,"w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(count)):
    s.writerow(count[i])
text.close()
