# DOCUMENTATION -------------------------------------------------------
'''
    https://techtutorialsx.com/2019/04/13/python-opencv-converting-image-to-black-and-white/
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
import csv
import matplotlib.pyplot as plt


# Keras 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D

# Project
import module0_sql as                   m0
import module1_img_prep as              m1
import module2_train_test_data_prep as  m2
import module3_models as                m3

# MYSQL CONNECTION -------------------------------------------------------

mydb    = mysql.connector.connect(
            host    ='localhost', 
            user    ='ccirelli2',
            passwd  ='Work4starr!', 
            database='GSU_SPRING_FP'
            )
mycursor    = mydb.cursor()

# LOAD DATA --------------------------------------------------------------

# Image files
dir_orig_imgs   = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/data/online/lineImages_all_original/lineImages'

dir_padded_imgs = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/data/online/Padded_imgs'


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
df_sample_handed    = m0.sql_query_random_sample_handedness(mydb, 500, 500).sample(frac=1)

# Generate Train / Test Split
X_train, y_train, X_test, y_test = m2.train_test_split(df_sample_handed, dir_padded_imgs)

# Reshape Data
X_train, y_train, X_test, y_test = m2.reshape_train_test_4_cnn(X_train, y_train, X_test, y_test)

# Run Model
def cnn_model(X_train, y_train, X_test, y_test, num_classes):
    '''
    Description:    
    Input_shape:    [row, col, dimensions]


    '''
    # Set Seed 
    seed    = 7
    np.random.seed(seed)

    # create model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(X_train.shape[1], X_train.shape[2], 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation= 'sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy' , optimizer='adam',
                  metrics=['accuracy'])

    # Fit Model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=5, batch_size=10)

    # Model Score
    score = model.evaluate(X_test, y_test, verbose=0)
    print(score)

    return model



cnn_model(X_train, y_train, X_test, y_test, 2)




















