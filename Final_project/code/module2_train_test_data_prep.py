# DOCUMENTATION ----------------------------------
'''
Description:        -   Generate query of random inputs.  So in the case of predicting 
                        gender we need to generate a random set of images with 50/50 
                        split of gender. 
                    -   We also need that set to contain the target variable, which 
                        needs to be appended to the dataset. 
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


# Function prepare Data
def create_nparray_imgs(df, dir_padded_files):
    '''
    Description:    return a list of imgs as numpy matrix
                    **Note that here is where we choose how to read the img
                    Ex Color or Gray scale
    Input           dataframe containing list of image file names.
    ''' 
    # Logging
    start = datetime.now()
    print('\nGenerating a list of images as numpy arrays')


    # Create List of Images
    list_imgs       = []
    list_handedness = [] 

    # Iterate File Names
    for afile in df.itertuples():
        
        # Create path 2 image
        path2file   = dir_padded_files + '/' + afile[1]
        
        # Read Image
        img_read    = cv2.imread(path2file, cv2.IMREAD_GRAYSCALE)
        
        # May result in null values as we removed larger images from dir
        if isinstance(img_read, np.ndarray) == True:
            # Attach image to list
            list_imgs.append(img_read)
            list_handedness.append(afile[-1])
   
    # Convert Lists to Numpy Arrays
    np_array_imgs       =   np.asarray(list_imgs)
    np_array_handedness =   np.asarray(list_handedness)

    # Logging
    end = datetime.now()
    print('Finished generating list.  List len => {}'.format(len(list_imgs)))
    print('Time to completion => {}\n'.format(start-end))

    # Return list (*Convert to numpy array for input to keras)
    return  np_array_imgs, np_array_handedness 





def train_test_split(df_sample_handed, dir_padded_files):
    '''
    Input:  df_sample_handed    = Dataframe including names of images
            dir_padded_files    = directory where padded files are located
    Output: Tuple X_train, y_train, X_test, y_test
    '''
    # Generate list of image as numpy arrays
    list_imgs, handedness = create_nparray_imgs(df_sample_handed, dir_padded_files)

    # Sample Proportions
    n_samples   = len(list_imgs)
    n_train     = int(round(0.7 * n_samples, 0))
    n_test      = n_samples - n_train
    
    # Convert Y To Binary Value
    dict_target = {'Right-handed':0, 'Left-handed':1}
    Y           = [dict_target[y] for y in handedness] 

    # Training Set 
    X_train     = list_imgs[0 : n_train]
    y_train     = Y[0 : n_train]

    # Test Set
    X_test      = list_imgs[n_train : ]
    y_test      = Y[n_train:  ]

    # Loging
    print('Completed Train Test split')
    print('Number of training images => {}'.format(len(X_train)))
    print('Number of test images     => {}\n'.format(len(X_test)))
  
    # Return
    return X_train, y_train, X_test, y_test


def reshape_train_test_4_seq_nn(X_train, y_train, X_test, y_test):
    # Logging
    print('Reshaping training & test data.  Flattening Arrays')

    # Calculate Num Pixels for Flattened Image row * column)
    num_pixels  = X_train[0].shape[0] * X_train[0].shape[1]

    # Flatten Images to Vectors 
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]
                ).astype('float32')
    X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]
                ).astype('float32')

    # Logging  
    print('Reshaping process finished.\nNew Dimension => 1 by {}'.format(X_train.shape))
    print('Converting target variable to categorical')

    # Normalize Pixel Sizes
    X_train     /=  255
    X_test      /=  255

    # Convert y to categorical
    y_train     = np_utils.to_categorical(y_train)
    y_test      = np_utils.to_categorical(y_test)

    # Num Classes
    num_classes = y_test.shape[1]

    return X_train, y_train, X_test, y_test



def reshape_train_test_4_cnn(X_train, y_train, X_test, y_test):
    ''' 
    Description:    Reshape train/test data w/ dims [samples][channels][width][height]
                    In  the case of a 2D image, the channel = 1
    Input:          train, test data
    Output:         train, test w/ new dims
    '''
    # Logging
    print('Reshaping feature data for CNN insertion')
    print('Original Shape training set => {}'.format(X_train.shape))

    # Reshape
    X_train_dim = X_train.shape
    X_test_dim  = X_test.shape
    X_train     = X_train.reshape(X_train_dim[0], X_train_dim[1], X_train_dim[2], 1
                                ).astype('float32')
    X_test      = X_test.reshape(X_test_dim[0], X_test_dim[1], X_test_dim[2], 1
                                ).astype('float32') 

    # Normalize Features
    X_train     = X_train/ 255 
    X_test      = X_test / 255 

    # Logging
    print('Feature reshaping completed')
    print('New X_train shape [num_imgs][row][col][dim]  => {}'.format(X_train.shape))
    print('New X_test shape  [num_imgs][row][col][dim]  => {}\n'.format(X_test.shape))
    print('Converting target to categorical variable')

    # Convert y to categorical
    y_train     = np_utils.to_categorical(y_train)
    y_test      = np_utils.to_categorical(y_test)

    # Number of classes
    num_classes = y_test.shape[1]

    # Logging
    print('Reshaping process completed')

    return X_train, y_train, X_test, y_test

