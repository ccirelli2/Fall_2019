# DOCUMENTATION -------------------------------------------------------
'''
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

# Project
import module1 as m1
import module0_sql as m0

# MYSQL CONNECTION ----------------------------------------------------
'''
mydb    = mysql.connector.connect(
            host    ='localhost', 
            user    =input('User => '),
            passwd  =input('Password => '), 
            database='GSU_SPRING_FP'
            )
mycursor    = mydb.cursor()
'''
# LOAD DATA -----------------------------------------------------------

# Writer Id Mapping
dir_output      = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/output'
writer_mapping  = r'mapping_writer_2_file.csv'
path2writerids  = dir_output + '/' + writer_mapping 

# Image files (ex: a02-108)
dir_original_imgs  = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/data/online/lineImages_all_original'

# Output directory for padded images
dir_output_new_imgs = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/data/online/lineImages_all_padded'



# FUNCTIONS -----------------------------------------------------------


def padd_imgs(dir_original_imgs, dir_output_new_imgs, print_img_shape=False, img_show=False):
    '''
    Input:      directory where images are located, directory to output padded images
    Output:     None.  We're going to save the padded images to a new directory.
                Padded images will need to have names that coincide with the 
                mapping to out authors.
    '''
    # List Full Path 2 Files
    list_paths2imgs = []
    
    # Define Designed Dimensions For Padded Image
    desired_rows    = 1400 
    desired_cols    = 3500

    # Template - Zero Vector Image w/ Desired Dimensions / Convert to White pixels
    np_array_zeros  = np.zeros((desired_rows, desired_cols))
    np_array_zeros[:,:]  = 255
    

    # Iterate Directory of Images
    for root, dirs, files in os.walk(dir_original_imgs):

        # Iterate list of directories in dirs
        for dir1 in dirs:
            # Define path 2 files or sub dir
            dir1_2imgs = root + '/' + dir1

            # List Files / Folders in Directory
            for img in os.listdir(dir1_2imgs):
        
                # If '.tif' in file/folder in directory
                if '.tif' in img:
                    # then we found the underlying files / append full path to list
                    path2img = dir1_2imgs + '/' + img
                    list_paths2imgs.append(path2img) 

                # If a directory, need to go one more down
                else:
                    # Define next full directory
                    dir2_2imgs  = dir1_2imgs + '/' + img
                    
                    # Iterate the next directory
                    for img in os.listdir(dir2_2imgs):
                        if '.tif' in img:
                            # Append full path 2 list
                            path2img = dir2_2imgs + '/' + img
                            list_paths2imgs.append(path2img)
    
    # Iterate List of Full Paths to Images
    for path2img in list_paths2imgs:

        
        # Open Image File
        original_img_open      = Image.open(path2img)
        original_img_array     = np.array(original_img_open)
        original_img_shape     = original_img_array.shape
       
        # Calculate Difference Between Actual and Zero Vector Image  
        row_diff        = desired_rows - original_img_shape[0]
        row_padd        = int(row_diff / 2)
        col_diff        = desired_cols - original_img_shape[1]
        col_padd        = int(col_diff / 2)

        if print_img_shape == True:
            print('Dimension small image => {}'.format(original_img_shape))
            print('Dimension big image   => {}'.format(np_array_zeros.shape))
            print('Row pad = {}, Column padd = {}'.format(row_padd, col_padd))


        # Assign values of original image to center of zero vector image
        '''
        Descriptions:   Think about the below as indexing a space in the middle
                        of your image with all zeros that is the same dimension
                        as your smaller images.  Then you just assigned that small
                        box the values of your smaller images
        '''
        np_array_zeros[row_padd: row_padd + original_img_shape[0], 
                       col_padd: original_img_shape[1] + col_padd] = original_img_array 
        
        # Show Image
        if img_show == True:
            img_padded      = cv2.imshow('Padded Image', np_array_zeros)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Write Images to File (padded image file name, padded image obj)
        filename = str(path2img.split('/')[-1].split('.')[0])
        cv2.imwrite(dir_output_new_imgs + '/' + filename + '.jpg', np_array_zeros)
        print('Padded image has been written to => {}'.format(dir_output_new_imgs + 
               '/' + filename + '.jpg'))



padd_imgs(dir_original_imgs, dir_output_new_imgs, False, False)






























































