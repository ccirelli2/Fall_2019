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
def create_list_imgs(df, dir_padded_files):
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

    # Iterate File Names
    for afile in df['full_name']:
        
        # Create path 2 image
        path2file   = dir_padded_files + '/' + afile
        
        # Read Image
        img_read    = cv2.imread(path2file, cv2.IMREAD_GRAYSCALE)
        
        # May result in null values as we removed larger images from dir
        if isinstance(img_read, np.ndarray) == True:
            # Attach image to list
            list_imgs.append(img_read)
    

    # Logging
    end = datetime.now()
    print('Finished generating list.  List len => {}'.format(len(list_imgs)))
    print('Time to completion => {}\n'.format(start-end))

    # Return list
    return list_imgs



def train_test_split(df_sample_handed, dir_padded_files):
    '''
    Input:  df_sample_handed    = Dataframe including names of images
            dir_padded_files    = directory where padded files are located
    Output: Tuple X_train, y_train, X_test, y_test
    '''
    # Generate list of image as numpy arrays
    list_imgs   = create_list_imgs(df_sample_handed, dir_padded_files)


    # Sample Proportions
    n_samples   = len(list_imgs)
    n_train     = int(round(0.7 * n_samples, 0))
    n_test      = n_samples - n_train

    # X Train / X Test
    X_train     = list_imgs[0 : n_train]
    X_test      = list_imgs[n_train : ]

    # Y Train / Y Test
    dict_target = {'Right-handed':0, 'Left-handed':1}
    Y           = [dict_target[y] for y in df_sample_handed['writing_hand']]
    y_train     = Y[0 : n_train]
    y_test      = Y[n_train :  ]

    # Loging
    print('Completed Train Test split')
    print('Number of training images => {}'.format(len(X_train)))
    print('Number of test images     => {}'.format(len(X_test)))

    # Return
    return X_train, y_train, X_test, y_test







def simple_NN(X_train, y_train, X_test, y_test):
    # Logging
    print('Building simple NN.  \nFlattening n-dim arrays')

    # Flatten MxN images to a vector for each image (row * column)
    num_pixels  = X_train[0].shape[1] * X_train[0].shape[2]

    print(X_train[0].shape)

    # Looks like we need to read the image in as a gray scale or B/W image
    '''https://techtutorialsx.com/2019/04/13/python-opencv-converting-image-to-black-and-white/'''

    '''
    # New Dataset
    X_train     = [x.reshape((1, num_pixels)) for x in X_train]
    y_train     = [y.reshape((1, num_pixels)) for y in y_train]
    
    X_test      = [x.reshape((1, num_pixels)) for x in X_test]
    y_test      = [y.reshape((1, num_pixels)) for y in y_test]
    
    # Logging  
    print('Reshaping process finished.\nNew Dimension => 1 by {}'.format(num_pixels))
    print(X_train[0].shape)    
    '''

