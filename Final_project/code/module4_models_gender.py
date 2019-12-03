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
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras import optimizers

# FUNCTION TO SAVE TRAINED MODELS ---------------------------------------

def cnn_model4_deeper_architecture(X_train, y_train, X_test, y_test, num_classes):
    ''' Description of Author's Model Architecture
    '''
    # Set Seed 
    seed    = 7
    np.random.seed(seed)

    # MODEL ARCHITECTURE -----------------------------------------
    model = Sequential()

    # 1st Convolution
    model.add(Convolution2D(240, kernel_size=(5, 5), input_shape=(X_train.shape[1], X_train.shape[2], 1),                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add Dropout
    model.add(Dropout(0.25))

    # 2nd Convolution
    model.add(Convolution2D(120, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Add Dropout
    model.add(Dropout(0.25))

    # 2nd Convolution
    model.add(Convolution2D(30, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Add Dropout
    model.add(Dropout(0.25))

    # Flatten
    model.add(Flatten())

    # Dense Layer1: Fully Connected Layers
    model.add(Dense(500, init='normal', activation='relu'))
    # Dense Layer2: Fully Connected Layers
    model.add(Dense(250, init='normal', activation='relu'))
    # Dense layer3: Final Activation Layer
    model.add(Dense(2, activation= 'softmax'))

    # Create Optimizer
    sgd     = optimizers.SGD(lr=0.001, decay=1e-7)

    # Compile model
    model.compile(loss='binary_crossentropy' , optimizer='adam',
                  metrics=['accuracy'])

    # Fit Model & Define History Object for Plotting
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        nb_epoch= 5, batch_size=5)

    print(history.history.keys())

    # Model Score
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Error rate => {}'.format((100-score[1]*100)))
    print(score)

    # Plot Accuracy History 
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()

    # Plot Loss History
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return model



