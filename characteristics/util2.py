import numpy as np
import cv2
from skimage.measure import regionprops
from scipy.spatial import distance
from scipy.signal import argrelextrema


def calculate_bvt(octa_image):
    # calculate blood vessel tortuosity
    skeleton = morphology.skeletonize(octa_image)
    distance_map = distance.squareform(distance.pdist(np.where(skeleton)))
    mean_distance = np.mean(distance_map)
    std_distance = np.std(distance_map)
    bvt = std_distance / mean_distance
    return bvt


def calculate_bvc(octa_image):
    # calculate blood vessel caliber
    skeleton = morphology.skeletonize(octa_image)
    endpoints = morphology.endpoints(skeleton)
    endpoint_positions = np.argwhere(endpoints)
    distances = distance.squareform(distance.pdist(endpoint_positions))
    max_distance = np.max(distances)
    bvc = max_distance / len(distances)
    return bvc


def calculate_bvd(octa_image):
    # calculate blood vessel density
    vessel_pixels = np.count_nonzero(octa_image)
    total_pixels = octa_image.shape[0] * octa_image.shape[1]
    bvd = vessel_pixels / total_pixels
    return bvd


def calculate_vpi(octa_image):
    # calculate vessel perimeter index
    skeleton = morphology.skeletonize(octa_image)
    endpoints = morphology.endpoints(skeleton)
    endpoint_positions = np.argwhere(endpoints)
    distances = distance.squareform(distance.pdist(endpoint_positions))
    max_distance = np.max(distances)
    vpi = np.sum(skeleton) / (2 * np.pi * max_distance)
    return vpi


def calculate_faz_ci(octa_image):
    # calculate FAZ contour irregularity
    threshold_value = cv2.threshold(octa_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[0]
    binary_image = (octa_image > threshold_value).astype(np.uint8)
    _, contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    faz_perimeter = cv2.arcLength(contours[0], True)
    max_length_index = argrelextrema(contours[0][:, :, 0], np.greater)[0][-1]
    min_length_index = argrelextrema(contours[0][:, :, 0], np.less)[0][0]
    length = distance.euclidean(contours[0][max_length_index][0], contours[0][min_length_index][0])
    faz_ci = (faz_perimeter**2) / (4 * np.pi * length)
    return faz_ci


def calculate_vci(octa_image):
    # calculate vessel complexity index
    props = regionprops(octa_image.astype(np.uint8))
    vci = props[0].perimeter / props[0].area
    return vci