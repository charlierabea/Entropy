import cv2
import numpy as np
import gabor as gabor

def do_otsu(img):
    im2 = img.copy()
    _, im2 = cv2.threshold(im2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return(im2)

def do_adaptive(img, kernel_size=5):
    im2 = img.copy()
    im2 = cv2.adaptiveThreshold(im2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY, kernel_size, 0)
    return(im2)

def do_GaussianBlur(gray):
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
    return blur_gray

def do_canny (im,t1=50,t2=150):
	"""
	This function simplifies the use
	of canny edge detector

	inputs:
		- im: OCT-A image
		- t1 and t2:  thresholds of canny edge detector
		- gamma: gamma parameter to canny edge detector
	"""
	im2 = im.copy()
	edges = cv2.GaussianBlur(im2, (15,15))
	edges = cv2.normalize(edges,edges,0,255,cv2.NORM_MINMAX)
	edges = cv2.Canny(np.uint8(edges),t1,t2)
	return edges

def do_gabor(img):
    return gabor.Gabor_process(img)

def do_erosiondilation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 100:
            cv2.drawContours(opening, [c], -1, (0,0,0), -1) 
    
    return opening