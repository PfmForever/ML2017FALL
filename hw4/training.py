from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional
from keras.layers.recurrent import GRU, LSTM
from keras.preprocessing import text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.embeddings import Embedding
from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing import text
import pickle
import sys
from gensim.models import Word2Vec as w2v
from keras.optimizers import Adam
from keras.layers import Activation
from keras import regularizers
from keras.layers.normalization import BatchNormalization
import csv
import numpy as np

# trainDataPath = sys.argv[1]
# testDataPath = Need Test
# trainDataNoLabelPath = sys.argv[2]

isSemi = False

trainDataPath = '../input_data/training_label.txt'
testDataPath = '../input_data/testing_data.txt'
trainDataNoLabelPath = '../input_data/training_nolabel.txt'
w2vDir = './w2v_100dim'
maxWordLen = 30
dropoutRate = 0.5

Model_filename = 'model.{epoch:02d}-{val_acc:.4f}.h5'
log_filename='log.csv'
# start process data
train_x_raw = []
train_y_raw = []
test_x_raw = []
semi_x_raw = []
with open(trainDataPath) as dataFile:
    inputLine = dataFile.readlines()
    for idx in range(len(inputLine)):
        inputLine[idx] = inputLine[idx].split('+++$+++')
        train_y_raw.append(inputLine[idx][0])
        train_x_raw.append(inputLine[idx][1])

# dont load the testing data to build ke_tokenizer cuz there's no testing data
# with open(testDataPath) as dataFile:
#     inputLine = dataFile.readlines()
#     for idx in range(len(inputLine)):
#         if(idx>=1):
#             test_x_raw.append((inputLine[idx].split(',', 1))[1].rstrip())

with open(trainDataNoLabelPath) as input_file:
    input_data = input_file.readlines()
    for i in range(len(input_data)):
        semi_x_raw.append(input_data[i].rstrip())


w2v_model = w2v.load(w2vDir)
weight = w2v_model.wv.syn0
num_word = weight.shape[0]
emb_dim = weight.shape[1]

with open('saved_tokenizer.pickle', 'rb') as input_tonz:
    tokenizer = pickle.load(input_tonz)

word2index = tokenizer.word_index
index2word = {v: k for k, v in word2index.items()}
embedding_matrix = np.ones((num_word,emb_dim))
for word, i in word2index.items():
    if i < num_word:
        try:
            embedding_matrix[i] = w2v_model.wv[word]
        except:
            embedding_matrix[i] = w2v_model.wv['i']
# build keras model
model = Sequential()
model.add(Embedding(num_word, emb_dim, weights=[embedding_matrix]))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(Bidirectional(LSTM(256)))
model.add(Dropout(dropoutRate))

model.add(Dense(256, activation='relu'))
model.add(Dropout(dropoutRate))
model.add(Dense(256, activation='relu'))
model.add(Dropout(dropoutRate))

model.add(Dense(512, activation='relu'))
model.add(Dropout(dropoutRate))
model.add(Dense(512, activation='relu'))
model.add(Dropout(dropoutRate))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(dropoutRate))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(dropoutRate))

model.add(Dense(2, activation='softmax'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
print ('model built')


train_sequences = tokenizer.texts_to_sequences(train_x_raw)
semi_sequences = tokenizer.texts_to_sequences(semi_x_raw)
for b in range(len(train_sequences)):
    for i in range(len(train_sequences[b])):
        try:
            w2v_model.wv[index2word[train_sequences[b][i]]]
        except KeyError:
            train_sequences[b][i] = 1

train_sequences = pad_sequences(train_sequences, maxlen=maxWordLen, value=1)
# print(train_sequences)

for b in range(len(semi_sequences)):
    for i in range(len(semi_sequences[b])):
        try:
            w2v_model.wv[index2word[semi_sequences[b][i]]]
        except KeyError:
            semi_sequences[b][i] = 1

semi_sequences = pad_sequences(semi_sequences, maxlen=maxWordLen, value=1)

y = to_categorical(train_y_raw)

cllbks = [
    CSVLogger(log_filename, append=True, separator=';'),
    # EarlyStopping(monitor='val_loss', patience=100, verbose=0),
    TensorBoard(log_dir='./Graph'),
    ModelCheckpoint(Model_filename, monitor='val_acc', verbose=1, save_best_only=False, mode='auto', period=1),
    # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100, verbose=1)
        ]
model.fit(train_sequences, y, batch_size=128, epochs=2, callbacks=cllbks, validation_split=0.07)


# semi-supervised training
if isSemi :
    semiModel_filename = 'Semi.{epoch:02d}-{val_acc:.4f}.h5'
    cllbks = [
    CSVLogger(log_filename, append=True, separator=';'),
    # EarlyStopping(monitor='val_loss', patience=100, verbose=0),
    TensorBoard(log_dir='./Graph'),
    ModelCheckpoint(semiModel_filename, monitor='val_acc', verbose=1, save_best_only=False, mode='auto', period=1),
    # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100, verbose=1)
        ]
    # repeat 10 times
    for i in range(10):
        # label the semi-data
        semi_pred = model.predict(semi_sequences)
        semi_ans = []
        for i in range(len(semi_pred)):
            semi_ans.append([str(i)])
            a = np.argmax(semi_pred[i])
            semi_ans[i].append(a)
        semi_Y = to_categorical(semi_ans)
        # train
        model.fit(semi_sequences, semi_Y,
                        epochs=2, 
                        batch_size=128,
                        callbacks=cllbks )