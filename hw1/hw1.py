import pandas
import numpy
# numpy.set_printoptions(threshold=numpy.nan)
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
data = numpy.empty((18,1))

for idx in range(0,240):
    data = numpy.concatenate((data,array[ 0 + (idx*18) : 18 + (idx*18) , 0  : 24 ]) , axis = 1)
data = numpy.delete( data , 0 , axis = 1)
data = data.astype(numpy.float)
# print(data[10,:])
# print(data[10,:])
# data.shape = (18, 5760)
#－－－－－－－－－－－－－－－－－－－－－－

#－－－－－－feature scaling－－－－－－
ptp = numpy.zeros((18,1))
ptp = numpy.ptp(data , axis = 1)
minimum = numpy.min(data,axis = 1)
# print(data[17,0])
for idx in range(0,17):
    data[idx,:] = data[idx,:]-minimum[idx]/ptp[idx]
# print(data.shape)
#－－－－－－－－－－－－－－－－－－－－－－