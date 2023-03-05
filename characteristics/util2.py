import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import regionprops
from skimage.morphology import skeletonize
from scipy.spatial import distance
from skimage.measure import label, regionprops

# Tang, F. Y., Ng, D. S., Lam, A., Luk, F., Wong, R., Chan, C., Mohamed, S., Fong, A., Lok, J., Tso, T., Lai, F., Brelen, M., Wong, T. Y., Tham, C. C., & Cheung, C. Y. (2017). Determinants of Quantitative Optical Coherence Tomography Angiography Metrics in Patients with Diabetes. Scientific reports, 7(1), 2575. https://doi.org/10.1038/s41598-017-02767-0

# raw
def calculate_vci(octa_image):
    '''
    Carpineto, P., Mastropasqua, R., Marchini, G., Toto, L., Di Nicola, M., & Di Antonio, L. (2016). 
    Reproducibility and repeatability of foveal avascular zone measurements in healthy subjects by 
    optical coherence tomography angiography. The British journal of ophthalmology, 100(5), 671–676. 
    https://doi.org/10.1136/bjophthalmol-2015-307330
    '''
    # calculate vessel complexity index
    props = regionprops(octa_image.astype(np.uint8))
    vci = props[0].perimeter / props[0].area
    return vci

#image_entropy: 
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

def calculate_bvd(octa_image):
    # calculate blood vessel density
    vessel_pixels = np.count_nonzero(octa_image)
    total_pixels = octa_image.shape[0] * octa_image.shape[1]
    bvd = vessel_pixels / total_pixels
    return bvd

#skeleton
def calculate_bvt(octa_image):
    # calculate blood vessel tortuosity
    skeleton = skeletonize(octa_image)
    distance_map = distance.squareform(distance.pdist(np.where(skeleton)))
    mean_distance = np.mean(distance_map)
    std_distance = np.std(distance_map)
    bvt = std_distance / mean_distance
    return bvt

def calculate_bvc(segmented_image):
    # create skeletonized vessel map
    skeleton = skeletonize(segmented_image)
    # calculate the numerator and denominator of the BVC equation
    numerator = np.sum(segmented_image)
    denominator = np.sum(skeleton)
    # calculate BVC
    bvc = numerator / denominator
    return bvc

#Vessel perimeter index
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

def fractal_dimension(Z):
    """Calculate the fractal dimension of an image."""
    # Only for 2d image
    assert(len(Z.shape) == 2)

    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # Count number of boxes with non-zero elements
        return len(np.where((S > 0) & (S < k*k))[0])

    # Transform Z into a binary array
    Z = (Z > 0).astype(np.uint8)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def calculate_fd(octa_image):
    """Calculate the fractal dimension of vessels in an OCT-A image."""
    # Binarize the image
    binary = (octa_image > 0).astype(np.uint8)
    # Label the connected components in the binary image
    labeled = label(binary)
    # Calculate the fractal dimension of each connected component
    fractal_dimensions = []
    for region in regionprops(labeled):
        if region.area > 10:  # Only consider regions with area > 10 pixels
            fractal_dimensions.append(fractal_dimension(region.image))
    # Calculate the mean fractal dimension of all connected components
    mean_fractal_dimension = np.mean(fractal_dimensions)
    return mean_fractal_dimension

# faz
def calculate_faz_ci(faz_image):
    try:
        # Convert to grayscale if necessary
        if faz_image.ndim == 3:
            faz_image = cv2.cvtColor(faz_image, cv2.COLOR_BGR2GRAY)

        # calculate FAZ contour irregularity
        size = faz_image.shape
        faz_area = cv2.countNonZero(faz_image)/(size[0]*size[1])

        # Calculate the perimeter of the FAZ
        contours, _ = cv2.findContours(faz_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(faz_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return 0, 0  # or some other default values
        faz_perimeter = cv2.arcLength(contours[0], True)

        # Calculate the FAZ circularity
        faz_ci = (4 * np.pi * faz_area) / (faz_perimeter ** 2)

        return faz_ci, faz_area
    except ZeroDivisionError:
        print("Error: division by zero")
        return None, None
