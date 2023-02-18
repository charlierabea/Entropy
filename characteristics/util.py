import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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

#skeletonize：感謝chatGPT
def do_skeletonize(img):
    im2 = img.copy()
    # Perform morphological operations to remove small objects or holes
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(im2, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Skeletonize the binary image
    skeleton = ximgproc.thinning(im2)
    
    return(skeleton)

#triangle：比otsu多一點
def do_findlarge(img):
    im2 = img.copy()
    # Perform morphological operations to remove small objects or holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opening = cv2.morphologyEx(im2, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 150:
            cv2.drawContours(opening, [c], -1, (0,0,0), -1)
                
    return(im2)




