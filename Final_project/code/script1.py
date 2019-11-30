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

# Image files
dir_orig_imgs   = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/data/online/lineImages_all_original/lineImages'

dir_2_padded_imgs = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/data/online/lineImages_all_padded'


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
df_sample_handed    = m0.sql_query_random_sample_handedness(mydb, 250, 250).sample(frac=1)

# Generate Train / Test Split
X_train, y_train, X_test, y_test = m2.train_test_split(df_sample_handed, dir_2_padded_imgs)


def simple_NN(X_train, y_train, X_test, y_test):
    # Logging
    print('Building simple NN.  \nFlattening n-dim arrays')

    # Calculate Num Pixels for Flattened Image row * column)
    num_pixels  = X_train[0].shape[0] * X_train[0].shape[1]

    # New Dataset
    X_train     = [x.reshape((1, num_pixels)) for x in X_train]     
    X_test      = [x.reshape((1, num_pixels)) for x in X_test]
     
    # Logging  
    print('Reshaping process finished.\nNew Dimension => 1 by {}'.format(num_pixels))
   
    


simple_NN(X_train, y_train, X_test, y_test)
