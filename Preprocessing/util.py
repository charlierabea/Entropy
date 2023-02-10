import cv2
import numpy as np
import gabor as gabor
import math

#otsu：找大血管
def do_otsu(img):
    im2 = img.astype('uint8')
    _, im2 = cv2.threshold(im2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return(im2)

#triangle：比otsu多一點
def do_triangle(img):
    im2 = img.astype('uint8')
    _, im2 = cv2.threshold(im2,0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
    return(im2)

#mean：Kernal 設大，外側只留大血管
def do_mean(img, kernel_size=333):
    im2 = img.copy()
    im2 = cv2.adaptiveThreshold(im2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY, kernel_size, 2)
    return(im2)

#gaussian：什麼都有
def do_gaussmean(img, kernel_size=11):
    im2 = img.copy()
    im2 = cv2.adaptiveThreshold(im2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, kernel_size, 2)
    return(im2)

#二值化後把小雜點去掉
def do_fillholes(img, kernal_size = (3,3), hole_area = 150):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernal_size)
    im2 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(im2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        area = cv2.contourArea(c)
        if hole_area < 150:
            cv2.drawContours(im2, [c], -1, (0,0,0), -1)
    
    return im2

#canny找邊緣
def do_canny (im,t1=50,t2=100):
	"""
	This function simplifies the use
	of canny edge detector

	inputs:
		- im: OCT-A image
		- t1 and t2:  thresholds of canny edge detector
		- gamma: gamma parameter to canny edge detector
	"""
	im2 = im.copy()
	edges = cv2.GaussianBlur(im2, (15,15), 0)
	edges = cv2.normalize(edges,edges,0,255,cv2.NORM_MINMAX)
	edges = cv2.Canny(edges,t1,t2)
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

def do_GaussianBlur(gray):
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
    return blur_gray
#想弄但弄不好的percentile跟moment
'''
def partial_sum(y, j):
    x = 0
    for i in range(j + 1):
        x += y[i]
    return x

def do_percentile(img):
    data = img.flatten()
    for i in range(len(data)):
        if data[i] > 0:
            maxbin = i

    for i in range(len(data) - 1, -1, -1):
        if data[i] > 0:
            minbin = i

    data2 = data[minbin:maxbin + 1]
    total = data2.sum()
    temp = float("inf")
    threshold = 0
    avec = [0] * 256
    for i in range(256):
        ptile = (i + 1) / 256.0
        avec[i] = abs((partial_sum(data2, i) / total) - ptile)
        if avec[i] < temp:
            temp = avec[i]
            threshold = i
    img[img > threshold] = 255
    img[img <= threshold] = 0
    return img

def do_Moments(data):
    total = 0
    m0 = 1.0
    m1 = 0.0
    m2 = 0.0
    m3 = 0.0
    sum = 0.0
    p0 = 0.0

    histo = np.zeros(data.shape)

    for i in range(data.shape[0]):
        total += data[i]

    for i in range(data.shape[0]):
        histo[i] = data[i] / total

    for i in range(data.shape[0]):
        m1 += i * histo[i]
        m2 += i * i * histo[i]
        m3 += i * i * i * histo[i]

    cd = m0 * m2 - m1 * m1
    c0 = (-m2 * m2 + m1 * m3) / cd
    c1 = (m0 * -m3 + m2 * m1) / cd
    z0 = 0.5 * (-c1 - np.sqrt(c1 * c1 - 4.0 * c0))
    z1 = 0.5 * (-c1 + np.sqrt(c1 * c1 - 4.0 * c0))
    p0 = (z1 - m1) / (z1 - z0)

    summation = 0
    threshold = -1
    for i in range(data.shape[0]):
        summation += histo[i]
        if summation > p0:
            threshold = i
            break

    return threshold
'''


