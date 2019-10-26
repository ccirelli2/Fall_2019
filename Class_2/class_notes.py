'''
INTRODUCTION TO OPENCV


Cons:
    -   If you find a solution for one image it is difficult to generalize to other images that
        may have different parameters.


Information:
    - somewhere between an intensity of 0 and 255. 

RGB:
    - OpenCV uses BGR.  So you need to tell it if you are looking at BGR. 


Other Color Spaces:  HSV

H = Hue, 0-359, is the primary color image. 
S = Saturation, 0-100%, represents how grey an image is 
V = Value, 0-100%, how bright 

Need to study the image. 

Each means a different channel or layer of the image. 
It is particular good for isolating colors. 

OpenCV cannot use a hue color because it passes 255 to 360.  
The benefit is that there is a one by one mapping between RGB and HSV. 

Presenter says that the utility of each is different basted on the application. 


Thresholding:
- Define a threshold value and set all pixel above all one color, and the remainder another. 
- In Range:  Set colors, like between A and B set the inside to a specific color. 
- Not useful to set global thresholds as it applies to the entire image.  You should tey "local thresholding". 
  To do this you need to define the locality. 
- Adaptive Thresholding:
    - Can use Mean or Gaussian. Gaussian is a smoothing function.  Mean takes the mean of all pixels in the kernel and 
      slides across. 
- Otsu thresholding ? automatically calculates a threshold value from image histogram. Only two peaks. 


Note:  Should always use threholding. 


Bluring:
- Smooths the image.
- Helps to generalize the image.  
- Without blurring your alogoritm will not generalize well. 
- Should be one of your first steps. 
** this is why likely our contours on the license plate were not good. 



Feature Extraction:
    - Trying to get the outline of an object. 
    - Separate sky, from elephant from trees.   Purpose of threholding and bluring is to be able to segment an image. 
    - Once you get segmentation you want to get the features. The features are what define the image. 
    - 


Corners:
    - Pass on to a diff algorithm that can analyze these corners. 
    - Algorithm sorts these images that can say these corners in these locations is a dog or person. 
    - 

Forground & Background:
    - depends on what your target is.
    - if a building, remove the background. 
    - Foreground removal removes your target. 
    - **Removing the foreground or background may help you to identify your target.
    - What is easiest to identify, the license plate or everything around it. 


Morphological Transformations:
    - Leverages kernels
    - M X N Kernel.  Usually equal size. 
    - Difference is that these kernels are filled with only 0's and 1's. 
    - Erosion:  Removes non-zero pixels. 
    - Dialation:  fattens lines




Harris Corner Detection:
    - Detects the corner no matter what direction you move in. 
    - If you see a change in any direction its a corner. 
    - 

? What is a determinant? 



Hough Transform:
    - a popular algorithm to detect lines and simple curves. 
    - How do you find the line in an image?  Hough helps you do this. 
    - produces a slope intercept for a single space. 


**boundingRect:
    - Allows you to bound your contours.  Puts a box around your contour. 
    - Returns the x,y coordinate of the top left corner of contour and 
      the widght and height. 


Cropping:
    - Use numpy.  Numpy uses y, x format. 


Fourier transform, super cool!!
- Remove high and low frequencies.  Derivative give you the frequency. 
- One to one map with frequency and spacial domain. 


'''


