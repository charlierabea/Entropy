#blood vessel tortuosity- Tang
# BVT is a measure of the degree of vessel distortion. In normal condition, the blood vessels transport blood efficiently, with a relatively smooth structure. However, in dis- eased conditions, the transportation efficiency of some blood vessels may be compromised due to distorted structure.

import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.spatial import distance_matrix

def binarize_image(image, threshold):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def extract_skeleton(binary_image):
    skeleton = skeletonize(binary_image / 255).astype(np.uint8)
    return skeleton

def find_endpoints(skeleton):
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    filtered = cv2.filter2D(skeleton, -1, kernel)
    endpoints = (filtered == 11).astype(np.uint8)
    return endpoints

def find_branches(skeleton, endpoints):
    labeled_endpoints = label(endpoints)
    endpoint_properties = regionprops(labeled_endpoints, cache=False)
    branch_points = []

    for ep in endpoint_properties:
        branch_points.append(np.array(ep.coords)[0])

    return branch_points

def adjacency_matrix(skeleton, branch_points):
    n = len(branch_points)
    adj_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = branch_points[i], branch_points[j]
            euclidean_dist = np.linalg.norm(p1 - p2)
            adj_matrix[i, j] = adj_matrix[j, i] = euclidean_dist * (skeleton[p1[0], p1[1]] or skeleton[p2[0], p2[1]])

    return adj_matrix

def calculate_bvt(skeleton, adjacency_matrix, branch_points):
    n = len(branch_points)
    total_ratio = 0

    for i in range(n):
        for j in range(i + 1, n):
            if adjacency_matrix[i, j] > 0:
                euclidean_dist = np.linalg.norm(branch_points[i] - branch_points[j])
                geodesic_dist = distance_matrix(branch_points[i][None], branch_points[j][None])[0, 0] * skeleton[branch_points[i][0], branch_points[i][1]]
                total_ratio += geodesic_dist / euclidean_dist

    return total_ratio / n

def calculate_bvt_from_image(image, threshold):
    #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    binary_image = binarize_image(image, threshold)
    skeleton = extract_skeleton(binary_image)
    endpoints = find_endpoints(skeleton)
    branch_points = find_branches(skeleton, endpoints)
    adj_matrix = adjacency_matrix(skeleton, branch_points)
    bvt = calculate_bvt(skeleton, adj_matrix, branch_points)

    return bvt

# # Example usage:
# image_path = 'path/to/image.png'
# threshold = 128  # Adjust the threshold value for your specific image
# bvt = calculate_bvt_from_image(image_path, threshold)
# print(f"Blood Vessel Tortuosity (BVT): {bvt}")

