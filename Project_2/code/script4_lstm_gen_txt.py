# DOCUMENTATION ----------------------------------------------------
'''
'''



# LOAD LIBRARIES ---------------------------------------------------

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
from keras.models import model_from_json

# Project 
import module2_lstm as m2



# LOAD DATA -----------------------------------------------------------

# Directories
dir_code    = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Project_2/code'
dir_output  = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Project_2/output'
dir_data    = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Project_2/data'

# Load File
afile       = r'M_fund.xlsx'
df          = pd.read_excel(dir_data + '/' + afile)
df_txt      = df['principal_strategies'].dropna().values



# PREPROCESS TEXT -----------------------------------------------------

# Clean Text (strip punctuation, use only dictionary words)
print('Removing punctuation')
punct           = string.punctuation + "0123456789-®—ø½§:|¡'`™"
txt_rm_punct    = [txt.translate({ord(i) : None for i in punct}) for txt in df_txt]
txt_joined      = ' '.join([txt for txt in txt_rm_punct])

# Create a Sample Set of Data
sample          = round(0.005 * len(txt_joined))
txt_sample      = txt_joined[: sample]
print('Length of sample text => {}'.format(len(txt_sample)))

# Enumerate Text
print('Creating Dictionary of Characters')
chars       = sorted(list(set(txt_sample)))
char_to_int = dict((c,i) for i, c in enumerate(chars))
int_to_char = dict((i,c) for i, c in enumerate(chars))

# CREATE TRAINING DATA
n_chars = len(txt_sample)
n_vocab = len(chars)

# Create Sequences 
seq_length  = 500
dataX       = []
dataY       = []
Count       = 0

# Iterate over entire text w/ a step 1.  Stop at n_chars minus seq length
print('Creating Sets of Sequences w/ Length of => {}'.format(seq_length))
for i in range(0, n_chars - seq_length, 1):
    seq_in  = txt_sample[i : i + seq_length] # this creates a seq of len 100
    seq_out = txt_sample[i + seq_length]     # this outputs the next char in the sequence
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out]) # does not require list comp as its a single char

n_patterns = len(dataX)

# Transform Data to Fit Sequence Required by Keras 
X = np.reshape(dataX, (n_patterns, seq_length, 1))

# Normalize X
X = X/ float(n_vocab)

# One Hot Encode The Output Variable
y = np_utils.to_categorical(dataY)



# LOAD MODEL ----------------------------------------------------------
print('Loading Pre-Trained Model & Weights')

# Saved Model Directory
dir_model = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Project_2/output/lstm_model_11162019'
# Load Weights
json_file           = open(dir_model + '/' + 'model_gen_txt_seq200.json', 'r')
loaded_model_json   = json_file.read()
json_file.close()
loaded_model        = model_from_json(loaded_model_json)

# Load Weights 
loaded_model.load_weights(dir_model + '/' + 'weights-improvement-20-0.7264.hdf5')
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam')


# MAKE PREDICTIONS ----------------------------------------------------

# Generate Random Seed 
''' Note that this pattern w/ have the same len as the one we generated 
    in the data prep stage'''
start           = np.random.randint(0, len(X)-1)
pattern_start   = dataX[start]

# Generate Predictions 
''' Iterate over number of predictions to generate (w/ gen indv chars)
    You will also need to decode the prediction as it will be an int. 
'''

def gen_prediction(model, n_predictions, pattern_start):

    # Pattern
    pattern = pattern_start
    pattern_decoded = ''.join([int_to_char[x] for x in pattern])
    print('\n\n*** START PATTERN ***', pattern_decoded)
    gen_txt = []

    # Iterate Range of Patterns to Generate
    for i in range(n_predictions):

        # reshape dim of input
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x/ float(n_vocab) 
        # generate prediction
        prediction = model.predict(x, verbose=0)
        # get the index of the large value
        index = np.argmax(prediction)
        # decode into to char
        result = int_to_char[index]
        gen_txt.append(result)
        #sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    print('\n\n*** GENERATED TEXT ***', ''.join(gen_txt)) 

gen_prediction(loaded_model, 1000, pattern_start)







































