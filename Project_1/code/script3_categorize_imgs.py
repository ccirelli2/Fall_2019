import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os


# IMPORT LICENSE PLATE IMAGES --------------------------------------------------------
os.chdir('/home/ccirelli2/Desktop/GSU/gsu_image_processing/Project_1/data/context')
list_imgs = os.listdir()



def iterate_imgs_manually_classify(list_imgs, write2excel = False):

    for num in range(0, 30):
            
        # Read Image
        img_read = cv2.imread(list_imgs[num])
        
        # Show Image
        cv2.imshow('Image', img_read)
        cv2.waitKey(0)

iterate_imgs_manually_classify(list_imgs)


def create_car_class_dataframe():

    #car_num             = [x for x in range(0,25)]
    class_background    = ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'D', 'D', 'L', 'L', 
                           'L', 'L', 'L', 'L', 'L', 'L', 'L', 'D', 'D', 'D', 'D', 'D']
    class_car           = ['D', 'L', 'L', 'D', 'D', 'L', 'D', 'D', 'L', 'D', 'D', 'D', 'D', 
                           'D', 'D', 'D', 'D', 'L', 'L', 'L', 'L', 'D', 'D', 'L', 'D'] 
    var      = [7417, 5383, 3611, 9181, 5990, 1978, 8927, 7423, 5823, 292, 161, 5540, 7490, 
                6980, 7340, 9260, 6852, 3611, 5573, 6581, 7674, 3749, 4184, 2290, 2755]
    mean     = [136, 141, 129, 149, 124, 139, 140, 153, 186, 19, 15, 131, 173, 150, 139, 173, 
                132, 182, 152, 135, 179, 116, 122, 67, 96]
      
    df = pd.DataFrame({})

    #df['car_num'] = car_num
    df['class_background'] = class_background
    df['class_car'] = class_car
    df['var'] = var
    df['mean'] = mean

    '''
    car_L = df['class_car'] == 'L'
    car_D = df['class_car'] == 'D'

    background_L = df['class_background'] == 'L'
    background_D = df['class_background'] == 'D'

    light_car = df[car_L]
    '''
    df_grouped = df.groupby(['class_background', 'class_car']).mean()

    print(df_grouped)
    



