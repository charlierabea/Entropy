import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#image entropy 
def image_entropy(image_path):
    # Open the image and convert it to grayscale
    with Image.open(image_path) as image:
        image = image.convert("L")
        # Convert the image to a numpy array
        im_arr = np.array(image)
        # Calculate the histogram of the image
        hist = np.histogram(im_arr, bins=256)[0]
        # Normalize the histogram
        hist = hist / float(im_arr.size)
        # Remove zeros from the histogram
        hist = hist[hist != 0]
        # Calculate the entropy
        entropy = -np.sum(hist * np.log2(hist))
        return entropy

#image vessel density(vd)
'''
Wang, X., Jiang, Y., Li, M., Zeng, W., & Zhou, Y. (2019). 
Automated measurement of retinal vessel density in optical 
coherence tomography angiography images using a vesselness 
filter. Journal of medical systems, 43(10), 327.
'''

def image_vd(image_path):
    # Open the image and convert it to grayscale
    # Load OCTA image
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    
    #Apply thresholding to segment blood vessels
    threshold = 127
    ret, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # Calculate vessel density as the proportion of non-zero pixels in the image
    vessel_density = np.count_nonzero(thresh) / (img.shape[0] * img.shape[1])

    return vessel_density
    
#image vessel length density(vld)
def calculate_vessel_length_density(octa_image, roi_mask):
    """
    Calculates vessel length density (VLD) from an OCTA image using a specified region of interest (ROI) mask.

    Args:
        octa_image: A 2D numpy array representing the OCTA image.
        roi_mask: A binary 2D numpy array representing the ROI mask.

    Returns:
        A float value representing the VLD within the specified ROI.
    """
    # Segment the blood vessels using a thresholding method
    threshold = 127
    vessel_mask = cv2.threshold(octa_image, threshold, 255, cv2.THRESH_BINARY)[1]
    vessel_mask = vessel_mask.astype(np.bool)

    # Apply the ROI mask to restrict the analysis to the specified region
    vessel_mask = np.logical_and(vessel_mask, roi_mask)

    # Calculate the total vessel length within the ROI
    skeleton = cv2.ximgproc.thinning(vessel_mask.astype(np.uint8))
    vessel_length = np.sum(skeleton)

    # Calculate the area of the ROI
    roi_area = np.sum(roi_mask)

    # Calculate the vessel length density
    vld = vessel_length / roi_area

    return vld

#fractal dimension

#fazarea

#fazcircularity

#vessel_diameter