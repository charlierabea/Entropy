import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import regionprops, label
from skimage.morphology import skeletonize, medial_axis
#from scipy.spatial import distance
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max

# Tang, F. Y., Ng, D. S., Lam, A., Luk, F., Wong, R., Chan, C., Mohamed, S., Fong, A., Lok, J., Tso, T., Lai, F., Brelen, M., Wong, T. Y., Tham, C. C., & Cheung, C. Y. (2017). Determinants of Quantitative Optical Coherence Tomography Angiography Metrics in Patients with Diabetes. Scientific reports, 7(1), 2575. https://doi.org/10.1038/s41598-017-02767-0

#image_entropy: 
# H = -Σ p_i * log2(p_i)
# C. E. Shannon, "A mathematical theory of communication," in The Bell System Technical Journal, vol. 27, no. 3, pp. 379-423, July 1948, doi: 10.1002/j.1538-7305.1948.tb01338.x.
def calculate_image_entropy(image_path):
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
    
# blood vessel density(BVD)- Tang
# reflects the ratio of the image area occupied by the blood vessels
# def calculate_bvd(octa_image):
#     # Calculate the sum of pixels occupied by vessels
#     vessel_pixels = np.sum(octa_image)
#     # Calculate the total number of pixels in the image
#     total_pixels = np.size(octa_image)
#     # Calculate the vessel density as the ratio of vessel pixels to total pixels
#     vessel_density = vessel_pixels / total_pixels
#     return vessel_density

def calculate_bvd(image_path):
    # Open the image and convert it to grayscale
    # Load OCTA image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    #Apply thresholding to segment blood vessels
    threshold = 127
    ret, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # Calculate vessel density as the proportion of non-zero pixels in the image
    vessel_density = np.count_nonzero(thresh) / (img.shape[0] * img.shape[1])

    return vessel_density

#vessel skeleton density(VSD)- Tang
def calculate_vsd(octa_image):
    # Create a skeletonized version of the OCTA image
    skeleton = skeletonize(octa_image)
    # Calculate the sum of pixels occupied by vessel skeleton
    skeleton_pixels = np.sum(skeleton)
    # Calculate the total number of pixels in the image
    total_pixels = np.size(octa_image)
    # Calculate the vessel skeleton density as the ratio of skeleton pixels to total pixels
    skeleton_density = skeleton_pixels / total_pixels
    return skeleton_density

#blood vessel tortuosity- Tang
# BVT is a measure of the degree of vessel distortion. In normal condition, the blood vessels transport blood efficiently, with a relatively smooth structure. However, in dis- eased conditions, the transportation efficiency of some blood vessels may be compromised due to distorted structure.
# def calculate_bvt(octa_image):
#     octa_image = cv2.cvtColor(octa_image, cv2.COLOR_BGR2GRAY)
#     # Skeletonize the binary OCTA image
#     skeleton, distance = medial_axis(octa_image, return_distance=True)
#     # Find the endpoints of the skeletonized branches
#     endpoints = peak_local_max(distance, min_distance=1, exclude_border=False)
#     # Calculate the number of branches
#     n_branches = len(endpoints) // 2
#     # Calculate the Euclidean distances between the endpoints of each branch
#     euclidean_distances = cdist(endpoints, endpoints)
#     np.fill_diagonal(euclidean_distances, np.inf)
#     euclidean_distances = np.min(euclidean_distances.reshape(n_branches, 2, 2), axis=1)
#     # Calculate the geodesic distances between the endpoints of each branch 
#     # (omitted for brevity)

#blood vessel calibre- Tang
# BVC, also named as vessel diameter, vessel width, or vessel diameter index, is used to quantify vascular dilation or shrinkage due to eye conditions
import numpy as np
from skimage.morphology import skeletonize

def calculate_bvc(segmented_image):
    # Create skeletonized vessel map
    skeleton = skeletonize(segmented_image)
    
    # Count the number of non-zero pixels (vessel spots) in the segmented image and skeleton
    vessel_count = np.count_nonzero(segmented_image)
    skeleton_count = np.count_nonzero(skeleton)

    # Calculate BVC
    if skeleton_count > 0:
        bvc = vessel_count / skeleton_count
    else:
        bvc = 0  # Avoid division by zero

    return bvc


#Vessel perimeter index(VPI)- Tang
# VPI measures the ratio between overall contour length of blood vessel boundaries and total blood vessel area in the segmented vessel map
def calculate_vpi(image):
    '''
    VPI measures the ratio between overall contour length of blood vessel boundaries and total blood vessel area in the segmented vessel map
    '''
    # Convert image to binary mask
    mask = np.where(image > 0, 1, 0).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate perimeter and area
    perimeter = np.sum([cv2.arcLength(contour, True) for contour in contours])
    area = np.sum(mask)

    # Calculate VPI
    vpi = perimeter / area

    return vpi
#fractal dimention(FD)- Tang
# Fractal dimension (FD) is a mathematical concept used to quantify the irregularity or complexity of patterns in an image. The fractal dimension is a non-integer value that can capture the degree of self-similarity and self-affinity of the pattern across different scales. High FD values indicate greater complexity and irregularity in the pattern, while low FD values indicate greater regularity and predictability.
def calculate_fd(octa_image):
    # Convert the binary OCTA image to a labeled image
    labeled_image = label(octa_image)
    # Extract the properties of each connected component
    regions = regionprops(labeled_image)
    # Find the largest connected component (i.e., the blood vessels)
    vessel_region = max(regions, key=lambda r: r.area)
    # Calculate the box-counting dimension of the blood vessels
    pixel_counts = np.histogram(vessel_region.filled_image, bins=2)[0]
    scales = np.logspace(np.log2(1), np.log2(vessel_region.area), num=50, base=2)
    n_boxes = []
    for scale in scales:
        n_boxes.append(count_boxes(vessel_region.filled_image, scale))
    n_boxes = np.array(n_boxes)
    fd, _ = np.polyfit(np.log(scales), np.log(n_boxes), deg=1)
    return fd

def count_boxes(image, scale):
    # Calculate the number of boxes needed to cover the image at the given scale
    if image.ndim == 3:
        image = image.min(axis=2)
    n_pixels = np.prod(image.shape)
    n_dim = int(np.sqrt(n_pixels))
    box_size = int(scale)
    n_boxes_row = n_dim // box_size
    n_boxes_col = n_dim // box_size
    boxes = np.zeros((n_boxes_row, n_boxes_col))
    for i in range(n_boxes_row):
        for j in range(n_boxes_col):
            box = image[i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size]
            boxes[i, j] = np.any(box)
    n_boxes = np.sum(boxes)
    return n_boxes
    
import numpy as np
import cv2

def calculate_fd2(octa_image):
    # Convert the image to grayscale if it's not already
    if len(octa_image.shape) == 3:
        octa_image = cv2.cvtColor(octa_image, cv2.COLOR_BGR2GRAY)
    
    # Binarize the image
    _, binarized = cv2.threshold(octa_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Define box sizes
    box_sizes = [2, 4, 8, 16, 32, 64]
    box_counts = []

    # For each box size, count how many boxes contain part of the image
    for size in box_sizes:
        count = 0
        for i in range(0, octa_image.shape[0], size):
            for j in range(0, octa_image.shape[1], size):
                if np.sum(binarized[i:i+size, j:j+size]) > 0:
                    count += 1
        box_counts.append(count)

    # Calculate the fractal dimension using a log-log plot
    log_box_sizes = np.log(1.0 / np.array(box_sizes))
    log_box_counts = np.log(box_counts)
    slope, _ = np.polyfit(log_box_sizes, log_box_counts, 1)
    
    return slope
    
# faz area/ faz contour irrelagularity(FAZ-CI)- Tang
# FAZ-CI measures the structural irregularity of the foveal shape
def calculate_faz_ci(faz_image):
    #faz_image = cv2.cvtColor(faz_image, cv2.COLOR_BGR2GRAY)
    # Label the binary FAZ image and extract the properties of each connected component
    labeled_image = label(faz_image)
    regions = regionprops(labeled_image)
    # Find the largest connected component (i.e., the FAZ)
    faz_region = max(regions, key=lambda r: r.area)
    # Calculate the perimeter of the FAZ
    faz_perimeter = faz_region.perimeter
    # Calculate the radius of a reference circle with area identical to the FAZ
    faz_area = faz_region.filled_area
    faz_radius = np.sqrt(faz_area / np.pi)
    # Calculate the perimeter of the reference circle
    reference_perimeter = 2 * np.pi * faz_radius
    # Calculate the FAZ-CI as the ratio of the FAZ perimeter to the reference circle perimeter
    faz_area = faz_area/ (faz_image.shape[0]*faz_image.shape[1])
    faz_ci = faz_perimeter / reference_perimeter
    return faz_area, faz_ci

# def calculate_faz_ci(faz_image):
#     try:
#         # Convert to grayscale if necessary
#         if faz_image.ndim == 3:
#             faz_image = cv2.cvtColor(faz_image, cv2.COLOR_BGR2GRAY)

#         # calculate FAZ contour irregularity
#         size = faz_image.shape
#         faz_area = cv2.countNonZero(faz_image)/(size[0]*size[1])

#         # Calculate the perimeter of the FAZ
#         contours, _ = cv2.findContours(faz_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contours, _ = cv2.findContours(faz_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if len(contours) == 0:
#             return 0, 0  # or some other default values
#         faz_perimeter = cv2.arcLength(contours[0], True)

#         # Calculate the FAZ circularity
#         faz_ci = (4 * np.pi * faz_area) / (faz_perimeter ** 2)

#         return faz_ci, faz_area
#     except ZeroDivisionError:
#         print("Error: division by zero")
#         return None, None


