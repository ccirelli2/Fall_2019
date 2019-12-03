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

def save_trained_model2file(trained_model, dir_output, model_name):
    '''Description:     saves trained model and weights to a json file
    '''
    
    # Change current directory to where you want to save the files
    os.chdir(dir_output)

    # Convert Model to Json Format
    model_json  = trained_model.to_json()

    # Define Model Name
    time_stamp  = str(datetime.now())[:-7].replace(' ','')
    model_name  = model_name + '_' + time_stamp
    weights_name= 'model_' + time_stamp + '.h5'

    # Write Model to File
    with open(model_name, 'w') as json_file:
        json_file.write(model_json)
    trained_model.save_weights(weights_name)
    print('Model and weights saved to => {}'.format(dir_output))



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


def cnn_model1(X_train, y_train, X_test, y_test, num_classes):
    '''
    Description: 1st model trained using the architecture from the keras 
                 tutorial. 
    Accuracy:    Seems to top out at 53%. 

    '''
    # Set Seed 
    seed    = 7
    np.random.seed(seed)

    # MODEL ARCHITECTURE -----------------------------------------
    model = Sequential()

    # 1st Convolution
    model.add(Conv2D(30, kernel_size=(5, 5), input_shape=(X_train.shape[1], X_train.shape[2], 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add Dropout
    model.add(Dropout(0.2))

    # 2nd Convolution
    model.add(Conv2D(15, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))


    # Flatten
    model.add(Flatten())

    # Add Dense Fully Connected Layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))

    # Sigmoid Function
    model.add(Dense(num_classes, activation= 'sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy' , optimizer='adam',
                  metrics=['accuracy'])

    # Fit Model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=2)

    # Model Score
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Error rate => {}'.format((100-score[1]*100)))
    print(score)

    return model



def cnn_model2(X_train, y_train, X_test, y_test, num_classes):
    '''
    Description:    Added more convolutional layers & dropouts 
    '''
    # Set Seed 
    seed    = 7
    np.random.seed(seed)

    # MODEL ARCHITECTURE -----------------------------------------
    model = Sequential()
    # 1st Convolution
    model.add(Conv2D(60, kernel_size=(7, 7), input_shape=(X_train.shape[1], X_train.shape[2], 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add Dropout
    model.add(Dropout(0.2))
    # 2nd Convolution 
    model.add(Conv2D(30, kernel_size=(5, 5), input_shape=(X_train.shape[1], X_train.shape[2], 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add Dropout
    model.add(Dropout(0.2))
    # 3rd Convolution
    model.add(Conv2D(15, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Add Dropout
    model.add(Dropout(0.2))
    # Flatten
    model.add(Flatten())
    # Add Dense Fully Connected Layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    # Sigmoid Function
    model.add(Dense(num_classes, activation= 'sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy' , optimizer=adagrad,
                  metrics=['accuracy'])
    # Fit Model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=5, batch_size=5)
    # Model Score
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Error rate => {}'.format((100-scores[1]*100)))
    print(score)
    return model

def cnn_model3_authors_architecture(X_train, y_train, X_test, y_test, num_classes):
    ''' Description of Author's Model Architecture
        
        1.) Model:          Convolutional Neueral Network
        2.) Layers:         6 layer architecture
                                2 stacks of convolutional 
                                2 subsampling (or max-pooilng) layers
                                2 final dense layers
        3.) Kernals:        5x5 for convolution layers
                            2x2 for pooling layers
                            *experiments showed that using smaller and larger kernels produced
                            worse results. 
        4.) Padding:        They use zero padding to preserve the spatial size
        5.) Activation:     All layers use ReLU and output layer uses SoftMax
        6.) Dropout:        Value of 0.25 was applied to each of the convolutional 
                            layers and with value of 0.5 to the first dense layer
        7.) Optimizer       Binary they used Stochastic Gradient Decent, 
                            Multiclass they used Adam. 
        8.) Learning Rate   Both used 0.001
                            Net weight decay of 1E-7
        9.) Input           Apparently the split the sentences into words
        10.) Preprocessing  Data Augmentation methods - normalization, rotation, 
                            shifting, and rescaling. 
        11.) Feature Maps   128 for the first convolutional layer, 256 for the second
        12.) Dense Layer    2 neurons for last layer.
        13.) Epochs         200
        14.) Data           100,000 synthetic training words, 20,000 validation words 
    '''
    # Set Seed 
    seed    = 7
    np.random.seed(seed)

    # MODEL ARCHITECTURE -----------------------------------------
    model = Sequential()

    # 1st Convolution
    model.add(Conv2D(64, kernel_size=(5, 5), input_shape=(X_train.shape[1], X_train.shape[2], 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add Dropout
    model.add(Dropout(0.25))

    # 2nd Convolution
    model.add(Conv2D(128, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
 
    # Add Dropout
    model.add(Dropout(0.5))

    # Flatten
    model.add(Flatten())

    # Dense Layer1: Fully Connected Layers
    model.add(Dense(200, activation='relu'))

    # Dense layer2: Final Activation Layer
    model.add(Dense(num_classes, activation= 'softmax'))

    # Create Optimizer
    sgd     = optimizers.SGD(lr=0.001, decay=1e-7)

    # Compile model
    model.compile(loss='binary_crossentropy' , optimizer=sgd,
                  metrics=['accuracy'])

    # Fit Model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=20, batch_size=10)

    # Model Score
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Error rate => {}'.format((100-score[1]*100)))
    print(score)

    return model


def cnn_model4_deeper_architecture(X_train, y_train, X_test, y_test, num_classes):
    ''' Description of Author's Model Architecture
    '''
    # Set Seed 
    seed    = 7 
    np.random.seed(seed)

    # MODEL ARCHITECTURE -----------------------------------------
    model = Sequential()

    # 1st Convolution
    model.add(Convolution2D(120, kernel_size=(5, 5), input_shape=(X_train.shape[1], X_train.shape[2], 1),                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add Dropout
    model.add(Dropout(0.25))

    # 2nd Convolution
    model.add(Convolution2D(60, kernel_size=(5,5), activation='relu'))
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
    model.add(Dense(120, init='normal', activation='relu'))
    # Dense Layer2: Fully Connected Layers
    model.add(Dense(60, init='normal', activation='relu'))
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




















