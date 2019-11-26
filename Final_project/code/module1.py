# DOCUMENTATION -------------------------------------------------------
'''
'''


# IMPORT LIBRARIES ----------------------------------------------------

# Python
import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from PIL import Image

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




