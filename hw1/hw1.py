import pandas
import numpy
import random
import math
import time
numpy.set_printoptions(threshold=numpy.nan)
import sys
#－－－－－載入csv並去除掉不需要的欄位－－－－－－
mydata = pandas.read_csv(sys.argv[1],encoding = 'big5')
mydata = mydata.drop('測站',axis=1)
mydata = mydata.drop('日期',axis=1)
mydata = mydata.drop('測項',axis=1)
mydata = mydata.replace(['NR'],[0])
# print(mydata)
#－－－－－－－－－－－－－－－－－－－－－－

#－－－－－－將資料整理成array，並把array設定成col=一小時資訊，row=某一測項－－－－－－
array = numpy.array(mydata)
trainData = numpy.empty((18,1))
# print(data)

for idx in range(0,240):
    trainData = numpy.concatenate((trainData,array[ 0 + (idx*18) : 18 + (idx*18) , 0  : 24 ]) , axis = 1)
trainData = numpy.delete( trainData , 0 , axis = 1)
trainData = trainData.astype(numpy.float)
pm2_5Ans = trainData[9,:]
#－－－－－－－－－－－－－－－－－－－－－－

#－－－－－－feature scaling－－－－－－
ptp = numpy.zeros((18,1))
ptp = numpy.ptp(trainData , axis = 1)
minimum = numpy.min(trainData,axis = 1)
for idx in range(0,17):
    trainData[idx,:] = trainData[idx,:]-minimum[idx]/ptp[idx]
#－－－－－－－－－－－－－－－－－－－－－－

#－－－－－－random select validation set－－－－－－
randomList = random.sample(range(0,239),24)
validSet = numpy.empty((18,0))
for idx in range(0,24): 
    validSet = numpy.concatenate((validSet,trainData[ 0 : 18 , 0 + idx*24 : 24 + idx*24]) , axis = 1)
    for idx_1 in range( 0 + idx*24 , 24 + idx*24):
        trainData = numpy.delete( trainData , idx_1 , axis=1)

# for idx in range(0,5183):
#     trainData[9,idx] = math.floor(trainData[9,idx])
# for idx in range(0,575):
#     pm2_5Ans[idx] = math.floor(pm2_5Ans[idx])
# trainData.shape = (18, 5184)
# validSet.shape = (18, 576)

#－－－－－－－－－－－－－－－－－－－－－－－－


#－－－－－－ExprotCSV function－－－－－－
def ExportCSV(array,name):
    df = pandas.DataFrame(array)
    df.to_csv(name+".csv")
#－－－－－－－－－－－－－－－－－－－－－－－－

#－－－－－－define parameter －－－－－－
b = 0
w = numpy.zeros((162,1))
lr = 0.00000000005
iteration = 1000000
pm2_5Ans = pm2_5Ans-0.0088495575221
for idx_1 in range(0,8):
    pm2_5Ans = numpy.delete( pm2_5Ans , idx_1)

#－－－－－－－－－－－－－－－－－－－－－－－－

# trainTemp = trainData[ :, 0:9 ]

# trainTemp = numpy.reshape(trainTemp,(162,1), order='F')
# print(trainTemp.shape)
# # print(trainTemp.shape)
# # print(trainTemp)
# predicTemp = b + numpy.dot(w , trainTemp)
# print(predicTemp)

#－－－－－－ start iteration －－－－－－
for idx in range(iteration):
    b_grad = 0.
    w_grad = numpy.zeros((162,1))
    loss = 0.
    history_w = numpy.array(w)
    for n in range(0,5174):
        trainTemp = trainData[ : , n : (n+9)]
        trainTemp = numpy.reshape(trainTemp,(162,1), order='F')
        predicTemp = b + numpy.vdot(w , trainTemp)
        b_grad = b_grad - 2.0 * (pm2_5Ans[n] - predicTemp )
        w_grad = w_grad - (2.0 * (pm2_5Ans[n] - predicTemp ) * trainTemp)
        loss = loss + (pm2_5Ans[n] - predicTemp)**2
    b = b - lr * b_grad
    w = w - lr * w_grad
    if(numpy.array_equal( w , history_w )):
        break
    print("Loss",loss)
    # print("now b = ")
    # print(b)
    # print("now w_grad[0] = ")
    # print(w_grad[0,0])    
    # print("now iteration = ")
    # print(idx)
    if(loss<1450000):
        lr = 0.0000000001
# #－－－－－－－－－－－－－－－－－－－－－－－－
print("final b = ")
print(b)
ExportCSV(w,"w_")