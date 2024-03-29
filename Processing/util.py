import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2.ximgproc as ximgproc
from skimage.morphology import skeletonize

#skeletonize：感謝chatGPT
def do_skeletonize(img):
    im2 = img.copy()
    _, im2 = cv2.threshold(im2, 0, 1, cv2.THRESH_BINARY)
    # Perform morphological operations to remove small objects or holes
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(im2, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Skeletonize the binary image
    #skeleton = ximgproc.thinning(im2)
    skeleton = skeletonize(im2)
    skeleton = skeleton*255
    
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
        if area < 500:
            cv2.drawContours(opening, [c], -1, (0,0,0), -1)
                
    return(im2)




