import numpy as np
import pandas
import numpy as np
import sys
import csv
import keras
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


x_test = Scaling(sys.argv[5]).values

model = keras.models.load_model('./bestmodel.h5')
predictions = model.predict(x_test)
res = pandas.DataFrame(predictions)
res = np.array(res)


ans = []

for i in range(res.shape[0]):
    ans.append([i+1])
    if(res[i,0]>res[i,1]):
        a = 0
    if(res[i,0]<res[i,1]):
        a = 1
    ans[i].append(a)
filename = sys.argv[6]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
