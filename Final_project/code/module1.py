# DOCUMENTATION -------------------------------------------------------
'''
Description:        Preprocess images

'''


# IMPORT LIBRARIES ----------------------------------------------------

# Python
import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from PIL import Image
import cv2

# Project
import module0_sql as m0


# FUNCTIONS ------------------------------------------------------------

def parse_author_info_xml_df(xml_file):
    '''
    Input:  Entire path to xml file
    ET      use to parse xml file into tree structure
    tag     will give you all the tags in the file
    attrib  attributes
    Example:
    {'name': '10208', 'DayOfBirth': '1971-06-03', 'EducationalDegree': 'Dipl. ing.', 
    'Gender': 'Male', 'NativeCountry': 'Switzerland', 'NativeLanguage': 'Swiss German', 
    'Profession': 'Student', 'Science': 'Computer Science', 'WritingType': 'Right-handed', 
    'WrittenLanguage': 'German'}
    '''
    # Parse XML File
    tree    = ET.parse(xml_file)
    root    = tree.getroot()
    
    # Create DataFrame of Data
    df = pd.DataFrame({})

    # List of Values
    name            = []
    dob             = []
    education       = []
    gender          = []
    country_origin  = []
    native_lang     = []
    profession      = []
    writing_hand    = []

    # Iterate Tree:
    for child in root:
        try:
            attrib      = child.attrib
            name.append(            attrib['name'])
            dob.append(             attrib['DayOfBirth'])
            education.append(       attrib['EducationalDegree'])
            gender.append(          attrib['Gender'])
            country_origin.append(  attrib['NativeCountry'])
            native_lang.append(     attrib['NativeLanguage'])
            writing_hand.append(    attrib['WritingType'])
        except KeyError as err:
            print('Key Error  => {}'.format(err))
            pass

    # Build DataFrame
    df['name']              = name
    df['dob']               = dob
    df['education']         = education
    df['gender']            = gender
    df['country_origin']    = country_origin
    df['native_language']   = native_lang
    df['writing_hand']      = writing_hand

    return df



# PARSE STROKE FILES

def parse_stroke_file(path2file):
    # Notes
    ''' 
    Input:      path2 xml file
    
    We'll need to create a huge dataframe with all of these values. 
        In addition to the stroke, we should capture the file name & author
    '''

    # Logging
    print('Parsing stroke file => {}'.format(path2file))

    # Parse XML File
    tree        = ET.parse(path2file)
    root        = tree.getroot()
    Count       = 0

    # Lists Capture Values
    form_id     = ''
    writer_id   = ''

    form_id_l   = []
    writer_id_l = []
    stroke      = []
    y           = []
    x           = []
    t           = []

    # Get Form Information
    for element in root.iter('Form'):
        form_id = element.attrib['id']
        writer_id = int(element.attrib['writerID'])

    # Iterate Each Stroke
    for element in root.iter('Stroke'):
        Count +=1
        # Iterate Element to Get Values
        for test in element.iter('Point'):
            # Build Lists of Values
            stroke.append(Count)
            form_id_l.append(form_id)
            writer_id_l.append(writer_id)
            y.append(int(test.attrib['y']))
            x.append(int(test.attrib['x']))
            t.append(float(test.attrib['time']))

    # Construct DataFrame
    df = pd.DataFrame({})
    df['form_id']   = form_id_l
    df['writer_id'] = writer_id_l
    df['stroke']    = stroke
    df['y']         = y
    df['x']         = x
    df['t']         = t

    # Logging
    print('Parsing completed. returning dataframe \n')

    return df





# GET MAXIMUM DIMENSION OF IMAGES 

def get_max_dim(dir_parent_images):
    '''
    Description:        Get Maximum dimension by iterating all images files. 
                        Purpose is to figure out how much padding is required
    '''
    # List Objects
    nrows   = []
    ncols   = []

    # Counter 
    Count = 0

    # Iterate Directory of Images
    for root, dirs, files in os.walk(dir_parent_images):
        for dir_ in dirs:
            dir2imgs   = root + '/' + dir_

            # Iterate Image Files In Each Subdirectory
            for img_n in os.listdir(dir2imgs):
                
                if '.' in img_n:
                    # Get Dimensions of Image
                    path2img    = dir2imgs + '/' + img_n
                    img_open    = Image.open(path2img)
                    img_array   = np.array(img_open) 
                    img_shape   = img_array.shape
                    
                    # Append Number of Rows & Columns
                    nrows.append(img_shape[0])
                    ncols.append(img_shape[1])
                    print('Image Shape {} by {}'.format(img_shape[0], img_shape[1]))
                    
                    # Logging
                    Count +=1
                    print('Count => {}'.format(Count))
        
    # Logging - Finish Obtaining Dimensions
    print('\n\n Finished obtaining dimensions of all image files')


    # Get Max Dimensions
    max_rows = max(nrows)
    max_cols = max(ncols)

    # Logging Max Values
    print('Maximum number of rows => {}'.format(max_rows))
    print('Maximum number of cols => {}'.format(max_cols))

    return nrows, ncols







def padd_imgs(list_path2imgs, dir_output_new_imgs, d_rows, d_cols, 
              print_img_shape=False, img_show=False):
    '''
    Input:      directory where images are located, directory to output padded images
    Output:     None.  We're going to save the padded images to a new directory.
                Padded images will need to have names that coincide with the 
                mapping to out authors.
    '''
    # Define Designed Dimensions For Padded Image
    desired_rows    = d_rows
    desired_cols    = d_cols

    # Template - Zero Vector Image w/ Desired Dimensions / Convert to White pixels
    np_array_zeros  = np.zeros((desired_rows, desired_cols))
    np_array_zeros[:,:]  = 255


    # Iterate List of Full Paths to Images
    for path2img in list_path2imgs:

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


        # END -----------------------------------------------









def get_all_files_in_dir(path):

    # Check if path is dir or file
    if '.' in path:
        #print(path)
        list_path2imgs.append(path)

    else:
        for adir in os.listdir(path):
            get_all_files_in_dir(path + '/' + adir)



def get_dict_freq_img_dims(get_all_files_in_dir):

    # Get list all files
    list_path2imgs = []
    get_all_files_in_dir(dir_orig_imgs)
   
    # Dict Object Hold Frequencies
    dict_row_dim_freq   = {}
    dict_col_dim_freq   = {}

    # DataFrame
    df_row = pd.DataFrame({})
    df_col = pd.DataFrame({})

    # Get Dimensions of Image
    for a_path2img in list_path2imgs:
        img_open    = Image.open(a_path2img)
        img_array   = np.array(img_open)
        img_shape   = img_array.shape

        # Generate Frequencies
        dict_row_dim_freq[img_shape[0]] = dict_row_dim_freq.get(img_shape[0], 1) + 1
        dict_col_dim_freq[img_shape[1]] = dict_col_dim_freq.get(img_shape[1], 1) + 1

    # Write Row Dim 2 File  
    df_row['Row_count'] = [x for x in dict_row_dim_freq]
    df_row['Frequencies'] = list(dict_row_dim_freq.values())
    df_row = df_row.sort_values(by= ['Row_count'])
    os.chdir(r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/output')
    df_row.to_excel('Row.xlsx')

    # Write Col Dim 2 File
    df_col['Col_count'] = [x for x in dict_col_dim_freq]
    df_col['Frequencies'] = list(dict_col_dim_freq.values())
    df_col = df_col.sort_values(by= ['Col_count'])
    os.chdir(r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/output')
    df_col.to_excel('Col.xlsx')

    # Logging
    print('finished')






def remove_img_files_with_mxn_dims(m, n, list_path2imgs):
    ''' 
    Description:    Remove imgs w/ dims that are outliers before setting
                    padding to images such that they all have the same dim
                    Otherwise, the padding will need to be around 
                    1200 x 3300 which will require significantly more 
                    computation time

    Imgs w/ rows    > 500
    Imgs w/ cols    > 2100
    '''
    # Count of files removed
    Count = 0 

    # Iterate List of Paths2 images
    for a_path2img in list_path2imgs:

        # Read Image & Get Shape
        img_open    = Image.open(a_path2img)
        img_array   = np.array(img_open)
        img_shape   = img_array.shape

        if img_shape[0] > m:
            os.remove(a_path2img)
            print('File Removed => {}'.format(a_path2img))
            Count +=1 
    
        elif img_shape[1] > n:
            os.remove(a_path2img)
            print('File Remove => {}'.format(a_path2img))
            Count +=1 

    # Final Log
    print('Number of Image Files Removed = {}'.format(Count))


    return None











