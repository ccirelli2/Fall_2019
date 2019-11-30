# DOCUMENTATION
'''
Description:        The purpose of this script is to downsize the original images, 
                    Then add padding to them such that they are all the same 
                    dimension for input into our models.

Step1               Get List of paths 2 all original images


'''

# Python
import pandas as pd
import numpy as np
import os
import sys
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
import module1_img_prep as m1


# MYSQL CONNECTION -------------------------------------------------------
'''
mydb    = mysql.connector.connect(
            host    ='localhost',
            user    ='ccirelli2',
            passwd  ='Work4starr!',
            database='GSU_SPRING_FP'
            )
mycursor    = mydb.cursor()
'''

# DATA --------------------------------------------------------------

# Image files
dir_orig_imgs   = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/data/online/lineImages_all_original/lineImages'
dir_resized_imgs= r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/data/online/Resized_imgs'
dir_output      = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/output' 

# RESHAPE IMAGES ------------------------------------------------------
# m1.reshape_original_images(dir_orig_imgs, dir_resized_imgs, 0.25, False)

# GET FREQUENCY OF ROW / COL DIMENSIONS -------------------------------
#m1.get_dict_freq_img_dims(dir_resized_imgs, dir_output)
'''
Description     Purpose is to preserve 90% of the dataset
Row Threshold   = 123
Col Threshold   = 535
'''

# PADD IMAGES & SAVE TO NEW DIRECTORY ----------------------------------
'''
m1.padd_imgs(dir_resized_imgs, 'Padded_imgs', d_rows=123, d_cols=535, print_img_shape=False, 
             img_show=False, mkdir=False)
'''













