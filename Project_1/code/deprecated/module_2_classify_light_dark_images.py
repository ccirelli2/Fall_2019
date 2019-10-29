import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt




def img_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey()

def img_gray(img):
    "You also add 0 to imread('name', 0)"
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def img_resize(img):
    dim = [600,600]
    return cv2.resize(img, (dim[0], dim[1]))


def show_n_images(list_imgs, num_imgs, grey = False):
    
    # Iterate over range 0 to num_imgs - 1
    for num in range(0, num_imgs):
        img = cv2.imread(list_imgs[num])
        # If selected grey images
        if grey == True:
            grey_image = img_grey(img)
            img_show(grey_image)
        # Otherwise display original
        else:
            img_show(img)
    # Return None
    return none



def gray_blur_img(img_raw):
    # Read Image
    img             = cv2.imread(img_raw)
    # Gray Image
    gray_img        = img_gray(img)
    # Blur Image
    'Blue:  last parameter set to 0 apparently finds the optimal solution'
    img_blur   = cv2.GaussianBlur(gray_img, (5,5), 0)
    # Return Image
    return img_blur

# Identify Dark and Light Images
def identify_light_dark_imgs(img, num, show_imgs = False):
    'Dimension = (height, weight, channels)'

    # Print Variance & Mean of Image
    gray_blurred_img    = gray_blur_img(img)
    mean                = gray_blurred_img.mean()
    var                 = gray_blurred_img.var()

    if show_imgs == True:
        print('Mean     = {}'.format(mean))
        print('Var      = {}'.format(var))
        cv2.imshow(gray_blurred_img)
        cv2.waitKey()

    return [mean, var]


def get_mean_var_imgs(list_imgs, num):
    car_dict = {}
    Car_intensity           = ['Dark',  'Light', 'Light', 'Dark',  'Dark',
                                'Light', 'Dark', 'Dark' , 'Light', 'Dark', 'Dark']
    Background_intensity    = ['Light', 'Light', 'Light', 'Light', 'Light', 'Light',
                                'Light','Light', 'Light', 'Dark', 'Dark']
    Mean                    = []
    Variance                = []
    for num in range(0,11):
        mean    = identify_light_dark_imgs(list_imgs[num], num)[0]
        var     = identify_light_dark_imgs(list_imgs[num], num)[1]
        Mean.append(mean)
        Variance.append(var)

    car_dict['Car'] = Car_intensity
    car_dict['Background'] = Background_intensity
    car_dict['Mean'] = Mean
    car_dict['Variance'] = Variance

    # Create DataFrame
    df = pd.DataFrame(car_dict)

    return df


      
       
# APPLY THRESHOLD -------------------------------------------------------------------

def classify_imgs_apply_threshold(img_raw, img_show_original = False):
    '''
    Puprose:    Classify Images by Car/Background mismatch and apply threshold  
    Process:    Read image
                Gray & Blur Image
                Classify Image
                Apply appropriate threshold
                Return image
    '''
    # Read Image
    img = cv2.imread(img_raw)
    # Gray Image
    gray = img_gray(img)
    # Blur Image
    'Blue:  last parameter set to 0 apparently finds the optimal solution'    
    img_gaus_blur = cv2.GaussianBlur(gray, (5,5), 0) 
    

    # Mean & Variance of Intensity - Sample 11 images 
    '''                         Mean     Variance
    Car         Background                           
    Dark        Dark            17.500017   227.159478
    Dark        Light           143.125681  7844.266154
    Light       Light           149.838849  4199.942653
    '''
    # Dark Car, Dark Background - No Threshold
    if img_gaus_blur.var() < 500 and img_gaus_blur.mean() < 50:
        # Don't apply a threshold operation
        if img_show_original == True:
            print('Dark Background & Dark Car - Apply No Threshold')
            cv2.imshow('Dark background, Dark car', img_gaus_blur)
            cv2.waitKey()
        # Return Image
        return (img_gaus_blur, 'Dark', 'Dark')

    # Light Car, Light Background - No Threshold
    elif img_gaus_blur.var() > 500 and img_gaus_blur.var() < 5500:

        # Darken Image & Apply Threshold
        img_darkened = cv2.add(img_gaus_blur, np.array([-100.0]))

        if img_show_original == True:
            print('Light Background & Light Car - Darken & Apply Threshold')          
            cv2.imshow('Image Darkened w/ Threshold', img_darkened)
            cv2.waitKey()

        # Return Image
        return (img_darkened, 'Light', 'Light')

    # Mismatch Car, Background - Apply Threshold
    elif img_gaus_blur.var() > 5500 and img_gaus_blur.mean() < 160:
        # Apply Threshold and return image
        img_add_threshold  = cv2.threshold(img_gaus_blur, 50, 255, 
                            cv2.THRESH_BINARY)

        # Show Image
        if img_show_original == True:
            print('Mismatch between car & background')
            cv2.imshow('Mismatch', img_gaus_blur)
            cv2.waitKey()
            # Show image with threshold applied 
            cv2.imshow('Binary Threshold Applied to Image', img_add_threshold[1])
            cv2.waitKey()
        # Return Image
        return (img_add_threshold[1], 'Mismatch', 'Mismatch')

    else:
        if img_show_original == True:
            print('Unknown')
            cv2.imshow('Unknown', img_gaus_blur)
            cv2.waitKey()
        # Return 
        return (img_gaus_blur, 'Unknown', 'Unknown')
  


