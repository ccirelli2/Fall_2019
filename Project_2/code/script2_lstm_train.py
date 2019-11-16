from __future__ import print_function

# DOCUMENTATION -----------------------------------------------------------
'''
Description:    Utilize LSTM model to learn to predict new text
                from teh principal strategies dataset. 

'''


# IMPORT LIBRARIES --------------------------------------------------------

# Python 
import pandas as pd
import numpy as np
import string
import os

import random
import io
import sys
import enchant
dict_en = enchant.Dict('en_US')


# Keras
from keras.preprocessing.text import Tokenizer as tk
from keras.preprocessing.text import text_to_word_sequence as ttws
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# Project 
import module2 as m2


# IMPORT DATA --------------------------------------------------------------

# Directories
dir_code    = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Project_2/code'
dir_output  = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Project_2/output'
dir_data    = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Project_2/data'

# Load File
afile       = r'M_fund.xlsx' 
df          = pd.read_excel(dir_data + '/' + afile)
df_txt      = df['principal_strategies'].dropna().values


# PREPROESSING TEXT --------------------------------------------------------

# Clean Text (strip punctuation, use only dictionary words)
print('Removing punctuation')
punct           = string.punctuation + "0123456789-®—ø½§:|¡'`™"
txt_rm_punct    = [txt.translate({ord(i) : None for i in punct}) for txt in df_txt]
txt_joined      = ' '.join([txt for txt in txt_rm_punct])

# Create a Sample Set of Data
'''Note that you could set this up in such a way that your sample comes from the rows
    of the dataframe'''
sample          = round(0.005 * len(txt_joined))
txt_sample      = txt_joined[: sample]
print('Length of sample text => {}'.format(len(txt_sample)))

# Enumerate Text
print('Creating Dictionary of Characters')
chars       = sorted(list(set(txt_sample)))
char_to_int = dict((c,i) for i, c in enumerate(chars))


# CREATE TRAINING DATA

n_chars = len(txt_sample)
n_vocab = len(chars)


# Create Sequences 
seq_length  = 100
dataX       = []
dataY       = []
Count       = 0

# Iterate over entire text w/ a step 1.  Stop at n_chars minus seq length
print('Creating Sets of Sequences')
for i in range(0, n_chars - seq_length, 1):
    seq_in  = txt_sample[i : i + seq_length] # this creates a seq of len 100
    seq_out = txt_sample[i + seq_length]     # this outputs the next char in the sequence
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out]) # does not require list comp as its a single char

n_patterns   = len(dataX)


# TRANSFORM DATA TO FIT SEQUENCE REQUIRED BY KERAS ----------------------------- 
X  np.reshape(dataX, (n_patterns, seq_length, 1))

# Normalize X
X = X/ float(n_vocab)

# One Hot Encode The Output Variable
y = np_utils.to_categorical(dataY)

# DEFINE LSTM MODEL -------------------------------------------------------------
print('Building model')
model   = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Define Checkpoints
filepath    = dir_output + '/' 'weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
checkpoint  = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='mim')
callbacks_list = [checkpoint]

# Fit Model:wq
print('Fitting Model')
model.fit(X, y, epochs=20, batch_size=200, callbacks=callbacks_list)


# SERIALIZE MODEL TO JSON -------------------------------------------------------
os.chdir(dir_output)
model_json  = model.to_json()
with open('model_gen_txt_seq200.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model_gen_txt_seq200.h5')
print('Model saved to disk')

























