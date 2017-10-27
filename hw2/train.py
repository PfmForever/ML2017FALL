import csv 
import pandas
import numpy as np
import sys
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import  Adam

def ExportCSV(array,name):
    df = pandas.DataFrame(array)
    df.to_csv(name+".csv")

def build_model():
    #建立模型
    model = Sequential()
    #將模型疊起
    model.add(Dense(input_dim=106,units=12,activation='sigmoid'))
    model.add(Dense(output_dim = 2,activation='sigmoid'))
    model.summary()
    return model
def stdScaling(arr,std,mean):
    arr = (arr-mean)/std
    return arr
def minMaxScaling(arr,ptp,mini):
    arr = (arr-mini)/ptp
    # print(arr)
    return arr

x_data_pd = pandas.read_csv('./data/X_train.csv')
y_data_pd = pandas.read_csv('./data/Y_train.csv')
x_test_pd = pandas.read_csv('./data/X_test.csv')
x_data = np.array(x_data_pd)
y_data = np.array(y_data_pd)
x_test = np.array(x_test_pd)

def Scaling(path):
    def dfScale(df):
        # convert object to float
        # df = df.apply(pandas.to_numeric)
        # mean-normalization
        # df=(df-df.mean())/df.std()
        # min-max normalization
        df=(df-df.min())/(df.max()-df.min())
        return df
    
    df = pandas.read_csv(path)
    # print(len(df.columns.values))
    continuous = ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']
    df_continuous = df[continuous]
    df_continuous = dfScale(df_continuous)
    for i in range(len(continuous)):
        del df[continuous[i]]
    
    df = pandas.concat([df_continuous, df], axis=1)
    # print(df)
    return df


xstd = np.array([np.std(x_data[:,0]) , np.std(x_data[:,1]) , np.std(x_data[:,3]) , np.std(x_data[:,4]) , np.std(x_data[:,5])])
xmean = np.array([np.mean(x_data[:,0]) , np.mean(x_data[:,1]) , np.mean(x_data[:,3]) , np.mean(x_data[:,4]) , np.mean(x_data[:,5])])

xptp = np.array([np.ptp(x_data[:,0],axis=0) ,np.ptp(x_data[:,1],axis=0) ,np.ptp(x_data[:,3],axis=0) ,np.ptp(x_data[:,4],axis=0) ,np.ptp(x_data[:,5],axis=0)])
xmin = np.array([x_data[:,0].min(axis = 0) ,x_data[:,1].min(axis = 0) ,x_data[:,3].min(axis = 0) ,x_data[:,4].min(axis = 0) ,x_data[:,5].min(axis = 0) ])

x_data = Scaling('./data/X_train.csv').values
x_test = Scaling('./data/X_test.csv').values
y_data = np_utils.to_categorical(y_data, 2)

(x_train,y_train)=(x_data,y_data)
model = build_model()
#開始訓練模型
model.compile(loss='binary_crossentropy',optimizer="rmsprop",metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=128,epochs=300)
#顯示訓練結果
score = model.evaluate(x_train,y_train)
print ('\nTrain Acc:', score[1])
predictions = model.predict(x_test)
model.save('./bestmodel.h5')
ExportCSV(predictions,'ressss')