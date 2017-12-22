import sys
import keras
import keras.backend as BK
from keras.models import Model, load_model
import numpy as np
import csv
import pickle
def rmse(y_true, y_pred):
    return BK.sqrt(BK.mean((y_pred-y_true) ** 2))

reader = csv.reader(open(sys.argv[1]))
test_data = list(reader)
test_data = np.array(test_data[1:], dtype = np.dtype('float64'))
model = load_model('./no_normalize_model.15-0.8473.h5', custom_objects = {'rmse':rmse})
ans = model.predict(np.hsplit(test_data[:,1:], 2))


count = 0
for i in ans:
    if np.isnan(i):
        count += 1
with open(sys.argv[2], 'w') as f:
    writer = csv.writer(f, lineterminator = '\n')
    writer.writerow(['TestDataID','Rating'])
    for i in range(len(ans)):
        writer.writerow([i+1, ans[i][0]])