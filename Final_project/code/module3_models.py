# DOCUMENTATION -------------------------------------------------------
'''
Description:    Module to house various models
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


# MODELS -----------------------------------------------------------------

def base_sequential_model(reshape_test_train_data):
    # Base line model
    model       = Sequential()
    model.add(Dense(num_pixels, input_dim = num_pixels, init='normal', activation='relu'))
    model.add(Dense(num_classes, init='normal', activation='sigmoid'))

    # Compile Model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit Model to Training Data 
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=1, batch_size=10)

    # Scores 
    scores  = model.evaluate(X_test, y_test, verbose=0)
    print(scores)














