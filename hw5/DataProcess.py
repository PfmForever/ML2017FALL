import pandas as pd
import numpy as np
from pandas import Series, DataFrame 
import pickle
trainData = './Data/train.csv'
userData = './Data/users.csv'
MOVIE_CSV = './Data/movies.csv'

reader = pd.read_csv(trainData)
x = np.array(DataFrame(reader,columns=['UserID','MovieID']).values.tolist())
y = np.array(DataFrame(reader,columns=['Rating']).values.tolist()).flatten()
y_mean = y.mean()
print(y)
print(y_mean)
y = y.astype(float)
y -= y_mean

pickle.dump(x, open('x.pkl', 'wb'))
pickle.dump(y, open('y.pkl', 'wb'))
pickle.dump(y_mean,open('y_mean.pkl','wb'))
