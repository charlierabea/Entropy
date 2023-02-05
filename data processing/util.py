import cv2
import numpy as np
import gabor as gabor

def do_otsu(img):
    im2 = img.astype('uint8')
    _, im2 = cv2.threshold(im2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return(im2)

def do_mean(img, kernel_size=5):
    im2 = img.copy()
    im2 = cv2.adaptiveThreshold(im2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY, kernel_size, 0)
    return(im2)

def do_gaussmean(img, kernel_size=5):
    im2 = img.copy()
    im2 = cv2.adaptiveThreshold(im2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, kernel_size, 0)
    return(im2)

'''
def partialSum(y, j):
    x = 0
    for i in range(j + 1):
        x += y[i]
    return x

def Percentile(data):
    ptile = 0.5
    avec = [0.0 for i in range(len(data))]

    total = partialSum(data, len(data) - 1)
    temp = 1.0
    for i in range(len(data)):
        avec[i] = abs((partialSum(data, i) / total) - ptile)
        if avec[i] < temp:
            temp = avec[i]
            threshold = i   

    return threshold

def do_percentile(img):
    im2 = img.copy()
    _, im2 = cv2.threshold(im2,Percentile(img),255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    return(im2)
'''

def partial_sum(y, j):
    x = 0
    for i in range(j + 1):
        x += y[i]
    return x

def do_percentile(img):
    data = img.flatten()
    total = data.sum()
    temp = float("inf")
    threshold = 0
    avec = [0] * 256
    for i in range(256):
        ptile = (i + 1) / 256.0
        avec[i] = abs((partial_sum(data, i) / total) - ptile)
        if avec[i] < temp:
            temp = avec[i]
            threshold = i
    img[img > threshold] = 255
    img[img <= threshold] = 0
    return img

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