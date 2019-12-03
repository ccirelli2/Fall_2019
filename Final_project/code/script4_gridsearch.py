# DOCUMENTATION -------------------------------------------------------
'''
    https://techtutorialsx.com/2019/04/13/python-opencv-converting-image-to-black-and-white/
    https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
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
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier

# Scikit Learn
from sklearn.model_selection import GridSearchCV

# Project
import module0_sql as                   m0
import module1_img_prep as              m1
import module2_train_test_data_prep as  m2
import module3_models_handedness as     m3
import module4_models_gender as         m4

# MYSQL CONNECTION -------------------------------------------------------

mydb    = mysql.connector.connect(
            host    ='localhost', 
            user    ='ccirelli2',
            passwd  ='Work4starr!', 
            database='GSU_SPRING_FP'
            )
mycursor    = mydb.cursor()

# LOAD DATA --------------------------------------------------------------

# Directories
dir_orig_imgs   = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/data/online/lineImages_all_original/lineImages'

dir_padded_imgs = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/data/online/Padded_imgs'
dir_trained_models  = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/trained_models'

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
df_sample_handed    = m0.sql_query_random_sample_gender(mydb, 50, 50).sample(frac=1)

# Generate Train / Test Split
X_train, y_train, X_test, y_test = m2.train_test_split(df_sample_handed, dir_padded_imgs, 
                                                        feature = -4)

# Reshape Data
X_train, y_train, X_test, y_test = m2.reshape_train_test_4_cnn(X_train, y_train, X_test, y_test)


# INSTANTIATE MODELS ------------------------------------------------------------- 

def create_model():
    ''' Description of Author's Model Architecture
    '''
    # MODEL ARCHITECTURE -----------------------------------------
    model = Sequential()

    # 1st Convolution
    model.add(Convolution2D(120, kernel_size=(5, 5), input_shape=(X_train.shape[1], 
            X_train.shape[2], 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add Dropout
    model.add(Dropout(0.25))

    # Flatten
    model.add(Flatten())

    # Dense Layer1: Fully Connected Layers
    model.add(Dense(120, init='normal', activation='relu'))
    # Dense layer: Final Activation Layer
    model.add(Dense(2, activation= 'sigmoid'))

    # Create Optimizer
    sgd     = optimizers.SGD(lr=0.001, decay=1e-7)

    # Compile model
    model.compile(loss='binary_crossentropy' , optimizer= sgd,
                 metrics=['accuracy'])

    return model


# DEFINE GRID SEARCH PARAMETERS --------------------------------------------

# Define Keras Classifier Model 
model   = KerasClassifier(build_fn  = create_model,verbose=0)

# Build Grid
''' n_jobs  = -1 / use cores in parallel
    cv      = cross validation
''' 
epochs      = [5]
batch_size  = [5,10]
param_grid  = dict(batch_size=batch_size, epochs=epochs)

# Instantiate Grid Search
grid        = GridSearchCV(estimator = model, param_grid=param_grid, n_jobs= 1, cv=2)
grid_result = grid.fit(X_train, y_train)

# Summarize REsults
print('Best {} using {}'.format(grid_result.best_score, grid_result.best_params_))
means   = grid_result.cv_results_['mean_test_score']
stdev    = grid_result.cv_results_['std_test_score']
params  = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{} {} with {}'.format(mean, stdev, param))





















