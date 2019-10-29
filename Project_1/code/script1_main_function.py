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


# MAIN FUNCTION _____________________________________________________________________

def main_match_contour_by_shape( template_img, target_img, num_template_contour, 
                            input_show_original_img         = False,
                            input_show_classified_img       = False, 
                            input_show_threshold_img        = False, 
                            input_show_edged_img            = False, 
                            input_show_all_target_contours  = False, 
                            input_draw_template_contours    = False, 
                            input_show_matching_contours    = True):
    '''
    template:       list_imgs[1]
    target:         all other images 
    num_template_
    contour:
                    the index value for the contour for which we are trying to match.      
    base_img:       This is the image that we are going to use to match against     
    test_img:       Image that we are going to test the algorithm to see if we can match 
                    the contour from the base image. 
    Operations:     1. Process images
                    2. Identify target contour
                    3. Try to match target contour on test image
                    4. Return original target image + convex hull to test match
    
                    For img0 = list_imgs[2], the target contour of the license plate from 
                    contours_sourted is [1] or the second in the list'''
    
    # Show Template & Target Images ---------------------------------------------------
    if input_show_original_img == True:
        m1.show_image(template_img, 'template img - original')
        m1.show_image(target_img, 'target img - original') 

    # Return Process Image Sorted Contours - Template & Target Images ------------------
    template_sorted_contours    = m1.process_img_and_return_sorted_contours(template_img, 
                'template image', input_show_classified_img, input_show_threshold_img, 
                                  input_show_edged_img)
    target_sorted_contours      = m1.process_img_and_return_sorted_contours(target_img, 
                'target image',   input_show_classified_img, input_show_threshold_img, 
                                  input_show_edged_img)
 
    # Show All Target Contours on Original Image ---------------------------------
    m1.show_all_target_contours(target_img, target_sorted_contours, input_show_all_target_contours) 

    # Define Template Contour (One we are trying to match) ------------------------
    template_contour            = template_sorted_contours[num_template_contour]
   
    # Show Template Contour On Original Image -------------------------------------
    if input_draw_template_contours == True:
        m1.draw_contours(template_img, template_contour, input_draw_template_contours)

    # Find Match Amongst Target Contours using Template (cv2.matchShapes) -------- 
    m1.find_matching_contour(target_img, template_contour, target_sorted_contours, 
                            input_show_matching_contours)


    # Return None ----------------------------------------------------------------
    return None



# RUN MAIN FUNCTION _____________________________________________________________

# Template Image
template_img    = list_imgs[2]

# Notes:
'''Try to pass the hull as opposed to the actual contour'''


for target_img in list_imgs:
    main_match_contour_by_shape(template_img, target_img, num_template_contour = 1, 
                            input_show_original_img         = False,
                            input_show_classified_img       = False, 
                            input_show_threshold_img        = False, 
                            input_show_edged_img            = False, 
                            input_show_all_target_contours  = True, 
                            input_draw_template_contours    = False, 
                            input_show_matching_contours    = False)
    cv2.destroyAllWindows()



'''
# Test Match Contour Algorithm on Same Images (results give a perfect match, so its working)
for num in range(0,40):
    print(num)
    main_match_contour_by_shape(template_img, target_img = list_imgs[2], 
                                num_template_contour = num, 
                                input_show_original_img         = False,
                                input_show_classified_img       = False, 
                                input_show_threshold_img        = False, 
                                input_show_edged_img            = False, 
                                input_show_all_target_contours  = False, 
                                input_draw_template_contours    = True, 
                                input_show_matching_contours    = False)
    cv2.destroyAllWindows()
'''




