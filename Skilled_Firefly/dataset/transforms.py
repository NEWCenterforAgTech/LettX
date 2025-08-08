"""Image transformation functions for preprocessing."""

import numpy as np
from PIL import Image
from functools import partial
from torchvision import transforms
from skimage.transform import resize

def hs_crop(image, crop_size=(224, 224), crop_mode='center'):
    """
    Takes a random or center crop from an image.

    Parameters:
        image (numpy.ndarray): The input image as a NumPy array of shape (h, w, c).
        crop_size (tuple): The desired crop size (height, width).
        crop_mode (str): The mode of cropping - 'random' or 'center'.

    Returns:
        numpy.ndarray: The cropped image.
    """
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size

    # Ensure the crop size is not greater than the image size
    if crop_h > h or crop_w > w:
        # Add padding if crop size is greater
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Pad the image with zeros
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                       mode='constant', constant_values=0)
        
        # Update dimensions after padding
        h, w = image.shape[:2]

    if crop_mode == 'random':
        # Choose the top-left corner randomly
        start_h = np.random.randint(0, h - crop_h + 1)
        start_w = np.random.randint(0, w - crop_w + 1)
    elif crop_mode == 'center':
        # Choose the center of the image
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
    else:
        raise ValueError("crop_mode must be either 'random' or 'center'.")

    # Perform the crop
    return image[start_h : start_h + crop_h, start_w : start_w + crop_w, :]

def center_square_crop_and_resize(img, new_hw=(512, 512)):
    """Numpy-based crop and resize for hyperspectral data"""
    h, w, c = img.shape
    
    # Create square crop (take center)
    crop_size = min(h, w)
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    crop = img[start_h:start_h+crop_size, start_w:start_w+crop_size, :]
    
    resized = resize(crop, (new_hw[0], new_hw[1], img.shape[2]), 
                    preserve_range=True, anti_aliasing=True)
    
    return resized

def hs_train_transforms(crop_size=(224, 224), flip_lr_prob=0.5):
    """
    Returns a composition of transforms to be applied for training images.
    
    Parameters:
        crop_size (tuple): Size of the crop (height, width)
        flip_lr_prob (float): Probability of horizontal flip
        
    Returns:
        transforms.Compose: Composed transformations
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=flip_lr_prob),
        transforms.RandomVerticalFlip(p=flip_lr_prob),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        partial(center_square_crop_and_resize, new_hw=crop_size),
    ])
    return transform

def hs_val_transforms(crop_size=(224, 224)):
    """
    Returns transforms to be applied for validation images.
    
    Parameters:
        crop_size (tuple): Size of the crop (height, width)
        
    Returns:
        list: List of transformations
    """
    #? Is this better: partial(center_square_crop_and_resize, new_hw=crop_size)
    transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
    ])
    return transform
    
def hs_test_transforms(crop_size=(224, 224)):
    """
    Returns transforms to be applied for test images.
    
    Parameters:
        crop_size (tuple): Size of the crop (height, width)
        
    Returns:
        list: List of transformations
    """

    return [partial(center_square_crop_and_resize, new_hw=crop_size),
            transforms.CenterCrop(crop_size)]

