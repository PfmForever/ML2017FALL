import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec as w2v
from keras.models import load_model
import csv
import numpy as np
import pickle

testDataPath = sys.argv[1]
predName = sys.argv[2]
modelName = './Final2_model.02-0.8174.h5'
w2vDir ='./w2v_100dim'
maxWordLen = 30

test_x_raw = []
with open(testDataPath) as dataFile:
    inputLine = dataFile.readlines()
    for idx in range(len(inputLine)):
        if(idx>=1):
            test_x_raw.append((inputLine[idx].split(',', 1))[1].rstrip())


with open('saved_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

test_sequences = tokenizer.texts_to_sequences(test_x_raw)

w2v_model = w2v.load(w2vDir)
word2index = tokenizer.word_index

index2word = {v: k for k, v in word2index.items()}

for b in range(len(test_sequences)):
    for i in range(len(test_sequences[b])):
        try:
            w2v_model.wv[index2word[test_sequences[b][i]]]
        except KeyError:
            test_sequences[b][i] = 1
            
test_sequences = pad_sequences(test_sequences, maxlen=maxWordLen, value=1)
model = load_model(modelName)
tst_predicted_out = model.predict(test_sequences)
print(tst_predicted_out)

ans = []
for i in range(len(tst_predicted_out)):
    ans.append([str(i)])
    a = np.argmax(tst_predicted_out[i])
    ans[i].append(a)

text = open(predName, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()


