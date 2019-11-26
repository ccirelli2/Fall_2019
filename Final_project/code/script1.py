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


# Project
import module1 as m1
import module0_sql as m0

# MYSQL CONNECTION ----------------------------------------------------
mydb    = mysql.connector.connect(
            host    ='localhost', 
            user    =input('User => '),
            passwd  =input('Password => '), 
            database='GSU_SPRING_FP'
            )
mycursor    = mydb.cursor()

# LOAD DATA -----------------------------------------------------------

# Author Info
dir_data    = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/data'
author_info = r'writers.xml'
path2_author= dir_data + '/' + author_info  


root_dir_data = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Final_project/data/online/original-xml-part/original'



# FUNCTIONS -----------------------------------------------------------







