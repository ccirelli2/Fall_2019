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
a = parse_author_info_xml(path2file)


print(a.head())







