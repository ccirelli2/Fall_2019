# DOCUMENTATION -------------------------------------------------------
'''
'''


# IMPORT LIBRARIES ----------------------------------------------------

# Python
import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET
import mysql.connector
from datetime import datetime
from PIL import Image
import cv2

# Keras 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils


# Project
import module0_sql as m0
import module1 as m1
import module2 as m2

# MYSQL CONNECTION -------------------------------------------------------

mydb    = mysql.connector.connect(
            host    ='localhost', 
            user    =input('User => '),
            passwd  =input('Password => '), 
            database='GSU_SPRING_FP'
            )
mycursor    = mydb.cursor()

# LOAD DATA --------------------------------------------------------------

# Writer Id Mapping
dir_output          = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/output' 
file_               = r'file_writer_img_mapping.csv'

# Image files
dir_padded_files    = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/data/online/lineImages_all_padded'


# DATASET METRICS ---------------------------------------------------------
'''
Description:        - Sql tables have list of file names and author info.
                    - Load random list of file names w/ even split right/left handed. 
                    - Need to write a function, read images in as numpy matrix, 
                      append to list.  
                    - Then we need to flatten the dimensions of each image to a 
                      1 dim array. 

'''
# Query Data / Randomly sort index (random sample, frac = 1 means return all)
df_sample_handed    = m0.sql_query_random_sample_handedness(mydb).sample(frac=1)

# Generate list of image as numpy arrays
list_imgs   = m2.create_list_imgs(df_sample_handed, dir_padded_files)    


'''
# Sample Proportions
n_samples   = len(df_sample_handed.index)
n_train     = int(round(0.7 * n_samples, 0))
n_test      = n_samples - n_train

# Train / Test Split
train       = df_sample_handed.loc[ : n_train]
test        = df_sample_handed.loc[n_train : ]


# Feature / Target Split
X_train     = train.loc[:, 'full_name': 'native_language']
y_train     = train['writing_hand']

y_test      = test['writing_hand']
X_test      = test.loc[ :, 'full_name': 'native_language']

# One Hot Encode Target
dict_target = {'Right-handed':0, 'Left-handed':1}
y_train     = [dict_target[y] for y in y_train]
y_test      = [dict_target[y] for y in y_test]


def simple_nn(x_train, y_train, x_test, y_test):
    seed = 7
    np.random(seed)


    pass
'''








































