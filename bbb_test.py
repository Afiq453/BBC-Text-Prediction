# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:40:12 2022

@author: AMD
"""

import pandas as pd
import os
import re
import json
import pickle
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Input
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Bidirectional,Embedding
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from modules import ModelCreation

#%% STATIC
PATH = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'

#EDA
#%% STEP 1) DATA LOADING
df=pd.read_csv(PATH)

#%% STEP 2) DATA INSPECTION

df.head(10)
df.tail(10)
df.info()
df.describe()

df['category'].unique() # to get the uniques targets
df['text'][0]
df['category'][0]

df.duplicated().sum() # 99 Duplicated data in this dataset
# CHECK Nan Value 
df.isna().sum() #check Nan Value

#%% STEP 3) DATA CLEANING

df = df.drop_duplicates() # REMOVE DUPLICATE

text = df['text'].values# features: x
category = df['category'].values # y


for index,rev in enumerate(text):
    # convert into lower case
    # remove numbers
    # ^ means not
    text[index] =re.sub('[^a-zA-Z]',' ',rev).lower()

#%% STEP 4) FEATURES SELECTION
# NOTHING TO SELECT

#%% STEP 5) PREPEOCESSING


#review = df['review'].values # features : X
#sentiment = df['sentiment'].values# sentiment : y

# 1) tokenization
vocab_size = 30000
oov_token = 'OOV'


tokenizer = Tokenizer(num_words = vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index
print(word_index)

train_sequences = tokenizer.texts_to_sequences(text) # to convert into numbers
# 2) Padding & truncating

length_of_text = [len(i) for i in train_sequences] #list comprehension
import numpy as np
print(np.mean(length_of_text)) # to get the number of max length for padding

max_len = 394


padded_review = pad_sequences(train_sequences,maxlen=max_len,
                              padding='post',
                              truncating='post')

#3) One hot encoding for the target


ohe = OneHotEncoder(sparse = False)
category = ohe.fit_transform(np.expand_dims(category, axis=-1))

#4) Train test split


X_train,X_test,y_train,y_test = train_test_split(padded_review,
                                                 category,
                                                 test_size=0.3,
                                                 random_state=123)

#%% Model development
# USE LSTM LAYERS, DROPOUT, DENSE, input

X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)

np.shape(X_train)[1:]
np.shape(category)[1] # (CHECK DENSE)


#model = Sequential ()
#model.add(Embedding(vocab_size,128))
#model.add(Bidirectional(LSTM(64,return_sequences=True)))
#model.add(Dropout(0.2))
#model.add(Bidirectional(LSTM(64)))
#model.add(Dropout(0.2))
#model.add(Dense(np.shape(category)[1],activation='softmax')) 
#model.summary()

Model = ModelCreation()
model = Model.simple_lstm_layer(vocab_size,np.shape(category)[1])
model.compile(optimizer='adam',
              loss = 'categorical_crossentropy',
              metrics=['acc'])


LOG_PATH = os.path.join(os.getcwd(),'logs')
log_dir = os.path.join(LOG_PATH, datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = EarlyStopping(monitor='loss', patience = 3)


hist = model.fit(X_train,y_train,
                 epochs=10,validation_data=(X_test,y_test),
                 callbacks=[tensorboard_callback,early_stopping_callback])


#%%

hist.history.keys()

plt.figure()
plt.plot(hist.history['loss'],label='Training Loss')
plt.plot(hist.history['val_loss'],label='Validation Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history['acc'],label='Training acc')
plt.plot(hist.history['val_acc'],label='Validation acc')
plt.legend()
plt.show()

#%% Model evaluation

y_true = y_test
y_pred = model.predict(X_test)

y_true = np.argmax(y_true,axis=1)
y_pred = np.argmax(y_pred,axis=1)


cr = classification_report(y_true, y_pred)
ac = accuracy_score(y_true, y_pred)
print(cr)
print('The accuracy_score is :',ac)

#%% MODEL SAVING

MODEL_SAVE_PATH = os.path.join(os.getcwd(),'saved_models','model.h5')
model.save(MODEL_SAVE_PATH)


token_json = tokenizer.to_json()
TOKENIZER_PATH = os.path.join(os.getcwd(),'saved_models','tokenizer_sentiment.json')
with open(TOKENIZER_PATH,'w') as file:
    json.dump(token_json,file)

OHE_PATH = os.path.join(os.getcwd(),'saved_models','ohe.pkl')
with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)
