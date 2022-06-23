# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:21:29 2022

@author: AMD
"""


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,BatchNormalization
from tensorflow.keras import Input

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Input
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Bidirectional,Embedding

class ModelCreation():
    def __init__(self):
        pass
    
    def simple_lstm_layer(self,vocab_size,num_shape):
        model = Sequential ()
        model.add(Embedding(vocab_size,128))
        model.add(Bidirectional(LSTM(64,return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.2))
        model.add(Dense(num_shape,activation='softmax')) 
        model.summary()
        
        return model

