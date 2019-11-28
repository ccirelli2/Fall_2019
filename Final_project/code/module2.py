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
    Input           dataframe containing list of image file names.
    '''
    # Logging
    print('Generating a list of images as numpy arrays')

    # Create List of Images
    list_imgs       = []

    # Iterate File Names
    for afile in df['full_name']:
        # Create path 2 image
        path2file   = dir_padded_files + '/' + afile
        # Read Image
        img_read    = cv2.imread(path2file)
        # Attach image to list
        list_imgs.append(img_read)

    # Logging
    print('Finished generating list.  List len => {}'.format(len(list_imgs)))

    # Return list
    return list_imgs

