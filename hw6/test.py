import csv
import pandas as pd
import keras
import sys
from keras.layers import Dense, Input
from keras.models import Model, load_model
import numpy as np
from sklearn.cluster import KMeans

x = np.load(sys.argv[1])
encoding_dim = 32
modelname = 'model.500-0.1845.h5'
test_data_path = sys.argv[2]
predictfile = sys.argv[3]

autoencoder = load_model(modelname)

input_img = Input(shape=(784,))

encoded = autoencoder.layers[1](input_img)
encoded = autoencoder.layers[2](encoded)

encoder = Model(input_img, encoded)


encoded_imgs = encoder.predict(x)

kmeans = KMeans(n_clusters=2)
kmeans.fit(encoded_imgs)
labels = kmeans.labels_

test = pd.read_csv(test_data_path, sep=',', header=0)
test = np.array(test.values)
pred = []
ans = []

for i in range(np.size(test,0)):
    ans.append([str(i)])
    f = lambda x: 1 if labels[test[i,1]] == labels[test[i,2]] else 0
    ans[i].append(f(i))

with open(predictfile, 'w') as f:
    writer = csv.writer(f, lineterminator = '\n')
    writer.writerow(['ID','Ans'])
    for i in range(len(ans)):
        writer.writerow(ans[i])