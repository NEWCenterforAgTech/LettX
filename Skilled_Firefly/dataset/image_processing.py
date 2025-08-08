"""Specialized image processing functions for hyperspectral data."""

import numpy as np
import cv2
import torch
from skimage.transform import resize

def remove_background(img):
    """
    Remove background from an image using NDVI thresholding.
    
    Parameters:
        img (np.ndarray): Input hyperspectral image
        
    Returns:
        np.ndarray: Image with background removed
    """
    ndvi_img = create_ndvi(img)
    mask_initial = ndvi_img > 0.35

    threshold_filter, centroid_list = get_largest_blobs(
        binary_image=mask_initial, num_blobs=1)
        
    filtered_image = img * threshold_filter[:, :, 0][:, :, np.newaxis]
    return filtered_image

def increase_brightness(img, value=80):
    """
    Increase the brightness of an image.
    
    Parameters:
        img (np.ndarray): Input image
        value (int): Brightness increase value
        
    Returns:
        np.ndarray: Brightened image
    """
    img = img.astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def fill_holes(binary_image):
    """
    Fill holes in a binary image using flood fill.
    
    Parameters:
        binary_image (np.ndarray): Binary image
        
    Returns:
        np.ndarray: Binary image with holes filled
    """
    h, w = binary_image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(255*binary_image.astype(np.uint8), mask, (0,0), 255)
    mask = cv2.bitwise_not(mask)
    return mask

def get_largest_blobs(binary_image, num_blobs):
    """
    Extract the largest blobs from a binary image.
    
    Parameters:
        binary_image (np.ndarray): Binary image
        num_blobs (int): Number of largest blobs to extract
        
    Returns:
        tuple: (Aggregate image containing blobs, List of blob centroids)
    """
    binary_image = binary_image.astype(np.uint8)
    num_comps, output, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8)

    sizes = stats[1:, -1]
    blob_indices = np.argsort(sizes)[::-1][0:num_blobs]

    aggregate_img = np.zeros((binary_image.shape[0], binary_image.shape[1], num_blobs))

    centroid_list = centroids[blob_indices+1]
    for i in range(num_blobs):
        img2 = np.zeros((output.shape))
        img2[output == blob_indices[i]+1] = 255
        img2 = fill_holes(img2) == 255
        aggregate_img[:,:,i] = img2[:-2,:-2]
    return aggregate_img, centroid_list

def create_rgb(hyper_cube):
    """
    Create an RGB image from a hyperspectral cube by selecting specific bands.
    
    Parameters:
        hyper_cube (np.ndarray or torch.Tensor): Hyperspectral image cube
        
    Returns:
        np.ndarray: RGB image
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(hyper_cube, torch.Tensor):
        hyper_cube = hyper_cube.cpu().numpy()
        
    rgb_image = hyper_cube[:, :, [49, 25, 12]]
    rgb_image = cv2.normalize(rgb_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    rgb_image = rgb_image.astype(np.uint8)
    # rgb_image = increase_brightness(rgb_image, value=80)
    return rgb_image

def create_ndvi(hyper_cube, epsilon=1e-10):
    """
    Create an NDVI (Normalized Difference Vegetation Index) image from a hyperspectral cube.
    
    # NDVI = (NIR - Red) / (NIR + Red)
    # NIR: Band 801.52435180022, Red: Band 601.04204336906
    
    Parameters:
        hyper_cube (np.ndarray): Hyperspectral image cube
        epsilon (float): Small value to prevent division by zero
        
    Returns:
        np.ndarray: NDVI image
    """
    band_nir = hyper_cube[:,:,81].astype(np.double)
    band_red = hyper_cube[:,:,40].astype(np.double)
    
    ndvi_image = np.true_divide((band_nir-band_red), (band_nir+band_red+epsilon))    
    return ndvi_image

def center_crop_and_resize(img, target_size=512):
    h, w = img.shape[:2]
    # First crop to square using the smaller dimension
    crop_size = min(h, w)
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    cropped = img[start_h:start_h+crop_size, start_w:start_w+crop_size, :]
    # Then resize (not crop) the square image to target size
    resized = resize(cropped, (target_size, target_size, img.shape[2]), 
                     preserve_range=True, anti_aliasing=True)
    return resized
