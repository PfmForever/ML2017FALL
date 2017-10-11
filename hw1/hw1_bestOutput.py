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
    for idx in range(0,36):
        inputData[ idx , : ] = ( inputData[ idx , : ] - minimum[idx] )/ptp[idx]
    return inputData
#－－－－－－－－－－－－－－－－－－－－－－
def TestDataScaling(inputData,ptp,min):
    for idx in range(0,36):
        # print("minimum[",idx,"]:",minimum[idx])
        # print("    ptp[",idx,"]:",ptp[idx])
        inputData[ : , 0+idx*9 : 9+idx*9 ] = (inputData[ : , 0+idx*9 : 9+idx*9 ]- minimum[idx])/ptp[idx]
    return inputData

# read model
w = np.load('./data/hw1_best_model.npy')
ptp= np.load('./data/hw1_best_ptp.npy')
minimum = np.load('./data/hw1_best_min.npy')

test_x = []
n_row = 0
text = open(sys.argv[1] ,"r")
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
# add square term
test_x = np.concatenate((test_x,test_x**2), axis=1)
test_x = TestDataScaling(test_x,ptp,minimum)
# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)



ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()