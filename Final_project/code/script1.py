# DOCUMENTATION -------------------------------------------------------
'''
'''


# IMPORT LIBRARIES ----------------------------------------------------

# Python
import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET


# Project


# LOAD DATA -----------------------------------------------------------


dir_data    = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/data'
afile       = r'writers.xml'
path2file   = dir_data + '/' + afile


def parse_author_info_xml(xml_file):
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



a = parse_author_info_xml(path2file)


print(a.head())







