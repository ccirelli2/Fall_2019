#DOCUMENTATION ------------------------------------------------------------------------- 
'''See "main_function_documentation.txt
'''


# IMPORT PYTHON PACKAGES ---------------------------------------------------------------
import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# IMPORT PROJECT MODULES --------------------------------------------------------------
import module1_main as m1

# IMPORT LICENSE PLATE IMAGES --------------------------------------------------------
os.chdir('/home/ccirelli2/Desktop/GSU/gsu_image_processing/Project_1/data/context')
list_imgs = os.listdir()


# RUN MAIN FUNCTION _____________________________________________________________

# Template Image
template_img    = list_imgs[2]


def get_contour_aread_single_template(template_img):
    'Template Contour Area = 573'
    img_car_read            = cv2.imread(template_img)
    img_car_gray            = cv2.cvtColor(img_car_read, cv2.COLOR_BGR2GRAY)
    img_car_blur            = cv2.bilateralFilter(img_car_gray, 11, 17, 17)
    edged = cv2.Canny(img_car_blur, 20, 200)
    contours, hierarchy     = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    n = len(contours)-1
    contours_sorted         = sorted(contours, key=cv2.contourArea, reverse=True)[:n]
    template_contour        = contours_sorted[4]
    m1.draw_contours(template_img, template_contour)
    contour_area            = cv2.contourArea(template_contour)
    print('Contour Area = {}'.format(contour_area))

def try_find_plate_using_contour_properties(list_imgs, show_all_contours = True):
    ''' Documentation: https://opencv-python.readthedocs.io/en/latest/doc/
                        17.imageContourProperty/imageContourProperty.html'''
    
    for image in list_imgs:
        img_car_read            = cv2.imread(image)
        img_car_gray            = cv2.cvtColor(img_car_read, cv2.COLOR_BGR2GRAY)
        img_car_blur            = cv2.bilateralFilter(img_car_gray, 11, 17, 17)
        edged = cv2.Canny(img_car_blur, 20, 200)
        cv2.imshow('Edged Image', edged)
        cv2.waitKey()
        contours, hierarchy     = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        n = len(contours)-1
        contours_sorted         = sorted(contours, key=cv2.contourArea, reverse=True)[:n]
        
        
        # Draw Contours  
        if show_all_contours == True:
            cv2.drawContours(img_car_read, contours_sorted, -1, (0,0,255), 3)
            cv2.imshow('Contours', img_car_read)
            cv2.waitKey(0)
         
        # Iterate contours
        for contour in contours_sorted:
            convex_hull         =       cv2.convexHull(contour) 
            convex_hull_area    =      cv2.contourArea(convex_hull)

            # Set Threshold
            if convex_hull_area > 500 and convex_hull_area < 1250:
                print('Convex Hull Area = {}'.format(convex_hull_area))
                # Generate Convex Hull & Display on Original Image
                m1.draw_contours(image, contour)

def get_bounded_rectangle(img):

    img_car_read            = cv2.imread(img)
    img_car_gray            = cv2.cvtColor(img_car_read, cv2.COLOR_BGR2GRAY)
    img_car_blur            = cv2.bilateralFilter(img_car_gray, 11, 17, 17)
    edged = cv2.Canny(img_car_blur, 20, 200)
    cv2.imshow('Edged Image', edged)
    cv2.waitKey()
    contours, hierarchy     = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    n = len(contours)-1
    contours_sorted         = sorted(contours, key=cv2.contourArea, reverse=True)[:n]
    
    for contour in contours_sorted:

        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        rect = cv2.rectangle(img_car_blur, (x,y), (x+w, y+h), (255,0,0), 3)
        print('Bounding Rectangle Area = {}'.format(area))
        cv2.imshow('Image', img_car_blur)
        cv2.waitKey(0)
        cv2.imshow('Rect', rect)
        cv2.waitKey(0)










