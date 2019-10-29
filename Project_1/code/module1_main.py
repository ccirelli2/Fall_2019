import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
import module2_classify_light_dark_images as m2

def gray_image(img):
    "You also add 0 to imread('name', 0)"
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def show_image(img_unread, img_name):
    img_read = cv2.imread(img_unread)
    cv2.imshow(img_name, img_read)
    cv2.waitKey(0)

def iterate_img_show_original(list_imgs):
    for num in range(0,10):
        img0 = list_imgs[num]
        img_read = cv2.imread(img0)
        cv2.imshow('image {}'.format(num), img_read)
        cv2.waitKey(0)

def draw_contours(template_img, template_contour, c_hull = True):
    ''' Note the difference between using a convex hull
        and the original contour'''
    
    template_img_read = cv2.imread(template_img)

    if c_hull == True:
        hull = cv2.convexHull(template_contour)
        cv2.drawContours(template_img_read, [hull], 0,(0,255,0), 2)
        cv2.imshow('Convex Hull', template_img_read)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        cv2.drawContours(img_read, [contour], 0,(0,255,0), 2)
        cv2.imshow('Original Contour', img_read)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def show_all_target_contours(target_img, target_sorted_contours, input_value):
    ''' input_value = true/false
        target_sorted_contours = list of sorted contours for target img'''
    if input_value == True:
        Count = 1 
        for contour in target_sorted_contours:
            print('Contour Count = {}'.format(Count))
            draw_contours(target_img, contour, c_hull = True)
            Count += 1

def classify_imgs_apply_appropriate_threshold(img_gaus_blur, img_show_original = False):
    'See module 2 for documentation'
    
    # Dark Car, Dark Background - No Threshold
    if img_gaus_blur.var() < 500 and img_gaus_blur.mean() < 50:
        # Lighten Image
        #img_lightened= cv2.add(img_gaus_blur, np.array([100.0]))
        img_add_threshold  = cv2.threshold(img_gaus_blur, 50, 255,
                            cv2.THRESH_BINARY)

        # Return Image
        return (img_add_threshold[1], 'Dark Car', 'Dark Background')

    # Light Car, Light Background - No Threshold
    elif img_gaus_blur.var() > 500 and img_gaus_blur.var() < 5500:
        # Darken Image & Apply Threshold
        img_darkened = cv2.add(img_gaus_blur, np.array([-75.0]))
        img_add_threshold  = cv2.threshold(img_darkened, 150, 255,
                            cv2.THRESH_TRUNC)
              
        # Return Image
        return (img_add_threshold[1], 'Light Car', 'Light Background')

    # Mismatch Car, Background - Apply Threshold
    elif img_gaus_blur.var() > 5500 and img_gaus_blur.mean() < 160:
        # Apply Threshold and return image
        img_add_threshold  = cv2.threshold(img_gaus_blur, 75, 255,
                            cv2.THRESH_BINARY)
        # Return Image
        return (img_add_threshold[1], 'Mismatch', 'Mismatch')

    # No Match Found - Return Original Blurred Image
    else:
        # Return
        return (img_gaus_blur, 'Unknown', 'Unknown')


def process_img_and_return_sorted_contours(img, img_name = None, 
                                show_classified_img   = False,   
                                show_threshold_img  = False, 
                                show_edged_img      = False):
    'This cannot be final function. Does not identify type of img before applying threshold'
    
    # Load Images ------------------------------------------------------------
    img_read = cv2.imread(img)
    
    # Convert to Gray
    img_gray = gray_image(img_read)

    # Blur
    img_blurred     = cv2.GaussianBlur(img_gray, (5,5), 0)

    # Classify Image & Apply Appropriate Threshold (see module 2 for documentation)
    img_classified  = classify_imgs_apply_appropriate_threshold(img_blurred)
    if show_classified_img == True:
        print('function working')
        cv2.imshow('{} - {} {} - Original'.format(img_name, 
                    img_classified[1], img_classified[2]), img_classified[0])
        cv2.waitKey(0)

    # Find Edges --------------------------------------------------------------
    '''(img, lower, upper):     lower and upper are the tresholds we apply to the
                                gradient intensity of img
                                Note:  Canny finds edges, now contours'''

    # Find Canny Edges --------------------------------------------------------
    img_edges = cv2.Canny(img_classified[0], 100, 255)
    if show_edged_img  == True:
        cv2.imshow('{} Edged'.format(img_name), img_edges)
        cv2.waitKey(0)
      
    
    # Find Contours -----------------------------------------------------------
    contours, hierarchy = cv2.findContours(img_edges, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
    # Sort Contours
    n = len(contours)-1
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:n]

    # Convert Contour to Convex_Hull
    convex_hull = [cv2.convexHull(x) for x in contours_sorted]

    # return sorted contours --------------------------------------------------
    return convex_hull


# Iterate through the contours and draw convex hull
def iteratively_show_convex_hull(contours, img_original):
    Count = 0
    for c in contours:
        # Print Count & Increase
        print('Count num = {}'.format(Count))
        Count += 1
        # Image Show
        hull = cv2.convexHull(c)
        print('*** HULL ***', hull)
        cv2.drawContours(img_original, [hull], 0 ,(0,255,0), 2)
        cv2.imshow('convex hull', img_original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Find Matching Contour 

def find_matching_contour(target_img, template_contour, target_sorted_contours, input_value):
    '''input_value:     Determines if we show the results or not'''

    # Show Results (T/F) --------------------------------------------------
    if input_value == True:
        
        # Count Contours For Target Image
        Count_contours = 0
        
        # Count Matches for Target Image
        Num_matchs = 0
        
        # Iterate Contours in Target Image --------------------------------
        for target_contour in target_sorted_contours:
            
            # Increase Contour Count
            Count_contours =+1

            # Get Height & Width of Contour 
            ''' 1.) We know the license plate length should be > its height. 
                2.) Standard license plate width 2x1.  So we can exclude 
                    aggregious outliers like 4x1'''
            x, y, w, h = cv2.boundingRect(target_contour)

            # Filter 1:  If Width > Height & width < 4 times height 
            if w > h and w < h*4:
                # Try to Match to the template contour ------------------------
                match = cv2.matchShapes(template_contour, target_contour, 2, 0.0)

                
                # Filter 2: If Error Rate < Threshold
                if match < 0.50:
                    Num_matchs += 1
                    print('Potential Match.  Contour Number = {}, Match Value = {}'.format(
                        Count_contours, match))
                    draw_contours(target_img, target_contour, True)
                
        # If No Potential Matches Found, Report None Found
        if Num_matchs == 0:
            print('No Potential Matches Found')








