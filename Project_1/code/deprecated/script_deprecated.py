#DOCUMENTATION ------------------------------------------------------------------------- 
'''
Thresholding:       https://docs.opencv.org/2.4/doc/tutorials/imgproc/threshold/
                    threshold.html?highlight=threshold

Application         If a pixel value is greater than a threshold value, it is assigned
                    one value, else another. 
        Syntax      cv2.threshold(img_gray, threshold_value, maxVal
        maxVal      I assume anything above the threshold gets converted to the maxVal
        Ex:         By setting our threshold to 75, which is fairly dark, any high int (white)
                    values will get onverted to white.
        Options     cv2.THRESH_TOZERO 
                    cv2.THRESH_TOZERO_INV
                    cv2.THRESH_BINARY
                    cv2.THRESH_BINARY_INV
                    cv2,THRESH_TRUNC
'''

# IMPORT PYTHON PACKAGES ---------------------------------------------------------------
import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# IMPORT PROJECT MODULES --------------------------------------------------------------
import module1 as m1

# IMPORT LICENSE PLATE IMAGES --------------------------------------------------------
os.chdir('/home/ccirelli2/Desktop/GSU/gsu_image_processing/Project_1/data/context')
list_imgs = os.listdir()


# CLASSIFY IMAGES  -------------------------------------------------------------------
'''
Function:           classify_imgs_apply_threshold
Description:        Code classifies images into three buckets 
                    1.) Dark Car, Dark Background
                    2.) Mismatch: Either Dark car light background, or Light car + dark background
                    3.) Light car, Light background
                    All images are converted to grey scale and blurred using a Gaussian kernel 
                    If a mismatch occurs, the binary threshold is applied. 
'''

def get_edges(img, show_img = True):
    edged = cv2.Canny(img[0], 25, 200) # (img, minVal, maxVal)
    if show_img == True:
        cv2.imshow('canny edges', edged)
        cv2.waitKey()
    return edged


def main_function(list_imgs):
    # Loop Over Images
    for num in range(0,20):
        # Classify Images (3 Categories)
        img_classified = m1.classify_imgs_apply_threshold(list_imgs[num], True)
        edged = get_edges(img_classified, True)

#main_function(list_imgs)



# WORKING WITH SINGLE IMAGE - GET CONTOURS AND ROI ------------------------------
img0 = list_imgs[2]
img_read = cv2.imread(img0)
img_gray = m1.img_gray(img_read)
img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
img_darkened = cv2.add(img_blur, np.array([-75.0]))
img_thresh   = cv2.threshold(img_darkened, 50, 255, cv2.THRESH_BINARY)
img_edges = cv2.Canny(img_thresh[1], 30, 200)

img_roi = img_contours[400:450, 400:500]
''' Url:        https://www.geeksforgeeks.org/find-and-draw-contours-using-opencv-python/
                https://circuitdigest.com/tutorial/image-segmentation-using-opencv
    Function    cv2.findContours(image, contour retrieval mode, 
                                contour approximation method, output of image
    Retrieval:  RETR_EXTERNAL
    Contour:    Numpy array (x,y) 
     
    Means of Storing Contours:
                CHAIN_APPROX_NONE:      Stores all boundary points. 
                CHAIN_APPROX_SIMPLE:    Provides start and end points.

    Hierarchy
                RETR_LIST:              All contours
                RETR_EXTERNAL:          external or outer contours
                RETR_CCOMP:             retrieves all in 2 lvl hierarchy
                RETR_TREE               retrieves all in full hierarchy. 

'''

# Get Contour 
contours, hierarchy = cv2.findContours(img_contours, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_NONE)

# Contour Area
'https://docs.opencv.org/trunk/d1/d32/tutorial_py_contour_properties.html'
area = cv2.contourArea(contours[100])

print(area)

'''
# Sort the contours by area then remove the largest frame contour
n = len(contours)-1
contours_sorted = sorted(contours, key=cv2.contourArea, reverse=False)[:n]


# Iterate through the contours and draw convex hull
for c in contours_sorted:
    hull = cv2.convexHull(c)
    cv2.drawContours(img_read, [hull], 0,(0,255,0), 2)
    cv2.imshow('convex hull', img_read)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''



#*** see matching contour by shapes ***https://circuitdigest.com/tutorial/image-segmentation-using-opencv


