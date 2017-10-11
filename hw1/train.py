import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import pandas

def ExportCSV(array,name):
    df = pandas.DataFrame(array)
    df.to_csv(name+".csv")

#－－－－－－feature scaling－－－－－－
def Scaling(inputData,ptp,min):
    for idx in range(0,18):
        inputData[ idx , : ] = ( inputData[ idx , : ] - minimum[idx] )/ptp[idx]
    return inputData
#－－－－－－－－－－－－－－－－－－－－－－
def TestDataScaling(inputData,ptp,min):
    for idx in range(0,18):
        inputData[ : , 0+idx*9 : 9+idx*9 ] = (inputData[ : , 0+idx*9 : 9+idx*9 ]- minimum[idx])/ptp[idx]
    # ExportCSV(inputData,"scalingTestData")
    return inputData


data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
	data.append([])

n_row = 0
text = open('./data/train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()
data = np.array(data)
scalingData = np.array(data)
ptp = np.zeros((1,18))
ptp = np.ptp(scalingData , axis = 1)
minimum = np.min(scalingData,axis = 1)
scalingData = Scaling(scalingData,ptp,minimum)
# ExportCSV(data,"data")

x = []
y = []
# 每 12 個月
for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        # 18種污染物
        for t in range(18):
            # 連續9小時
            for s in range(9):
                x[471*i+j].append(scalingData[t][480*i+j+s] )
        y.append(data[9][480*i+j+9])
x = np.array(x)
y = np.array(y)
# add square term
# x = np.concatenate((x,x**2), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

# print(x)
# ExportCSV(x,"x")


w = np.zeros(len(x[0]))
l_rate = 10
repeat = 20000

x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

# print(x.shape[0])

for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo - y

    cost = np.sum(loss**2) / len(x)
    cost_a  = math.sqrt(cost)
    
    gra = np.dot(x_t,loss)
    # print(gra.shape)
    s_gra += gra**2
    # print(s_gra.shape)
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada
    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))


# save model
np.save('./data/hw1_model.npy',w)
np.save('./data/hw1_ptp_min.npy',[ptp,minimum])
# # read model
# w = np.load('sample_model.npy')


test_x = []
n_row = 0
text = open('./data/test.csv' ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        for i in range(2,11):
            test_x[n_row//18].append(float(r[i]) )
    else :
        for i in range(2,11):
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)
# ExportCSV(test_x,"text_x")
# add square term
# test_x = np.concatenate((test_x,test_x**2), axis=1)
test_x = TestDataScaling(test_x,ptp,minimum)
# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)



ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

filename = "predict.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()