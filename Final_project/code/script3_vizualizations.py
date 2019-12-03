# DOCUMENTATION ------------------------------------------
'''
Descriptions        Functions to generate vizualizations for the final presenation

'''

# IMPORT LIBRARIES ---------------------------------------

# Python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model




# DIRECTORIES ---------------------------------------------

dir_orig_imgs   = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/data/online/lineImages_all_original/lineImages'
dir_saved_model = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/trained_models'


# FUNCTIONS -----------------------------------------------


# Generate A Frequency Plot of Image Dimensions

def get_dict_freq_img_dims(dir_orig_imgs, plot):

    # Logging
    print('Generating a Frequency Table for Dimensions of Images')

    # Get list all files
    list_path2imgs = []
    def get_all_files_in_dir(path):

        # Check if path is dir or file
        if '.' in path:
            #print(path)
            list_path2imgs.append(path)

        else:
            for adir in os.listdir(path):
                get_all_files_in_dir(path + '/' + adir)

    get_all_files_in_dir(dir_orig_imgs)

    # Dict Object Hold Frequencies
    dict_row_dim_freq   = {}
    dict_col_dim_freq   = {}

    # DataFrame
    df_row = pd.DataFrame({})
    df_col = pd.DataFrame({})

    # Get Dimensions of Image
    for a_path2img in list_path2imgs[:1000]:
        img_open    = Image.open(a_path2img)
        img_array   = np.array(img_open)
        img_shape   = img_array.shape

       # Generate Frequencies
        dict_row_dim_freq[img_shape[0]] = dict_row_dim_freq.get(img_shape[0], 1) + 1
        dict_col_dim_freq[img_shape[1]] = dict_col_dim_freq.get(img_shape[1], 1) + 1

    # Plot Row Info
    if plot == 'Row':
        df_row['Row_count'] = [x for x in dict_row_dim_freq]
        df_row['Frequencies'] = list(dict_row_dim_freq.values())
        df_row = df_row.sort_values(by= ['Row_count'])
        # Plot 
        df_row['Row_count'].plot(kind='hist', grid=True, fontsize=18 )
        plt.title('Image Row Count Distribution', fontsize=22)
        plt.ylabel('Frequency', fontsize=20)
        plt.xlabel('Row Count', fontsize=20)
        plt.show()

    # Plot Column Info
    elif plot == 'Col':
        df_col['Col_count'] = [x for x in dict_col_dim_freq]
        df_col['Frequencies'] = list(dict_col_dim_freq.values())
        df_col = df_col.sort_values(by= ['Col_count'])
        # Plot 
        df_col['Col_count'].plot(kind='hist', grid=True, fontsize=18 )
        plt.title('Image Col Count Distribution', fontsize=22)
        plt.ylabel('Frequency', fontsize=20)
        plt.xlabel('Col Count', fontsize=20)
        plt.show()

    # Logging
    print('finished')




# Vizualize Model 


def load_model(dir_saved_model):

    # change directory 
    os.chdir(dir_saved_model)

    # Load File
    json_file       = open('CNN_2019-12-0119:00:54.json', 'r')
    loaded_model_json   = json_file.read()
    json_file.close()
    loaded_model        = model_from_json(loaded_model_json)

    # Load Weights
    loaded_model.load_weights('model_2019-12-0119:00:54.h5')

    # Vizualize
    plot_model(load_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
load_model(dir_saved_model)




















