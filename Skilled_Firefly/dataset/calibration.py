"""Functions for calibrating hyperspectral images."""

import numpy as np
import pandas as pd

def read_calib(calib_path):
    """Read calibration data from a CSV file."""
    return pd.read_csv(calib_path)

def apply_calib(image, calib_data):
    """
    Apply calibration to an image using the formula:
    Calibrated_image = (Raw HSI Image - Black Reference) / (White Reference - Black Reference)
    """
    calib_values = calib_data.values
    black_ref = calib_values[0]
    white_ref = calib_values[1]
    calibrated_image = (image - black_ref) / (white_ref - black_ref)
    return calibrated_image

def find_closest_wavelength(wavs, target):
    """Find the index of the closest wavelength to the target."""
    return (np.abs(np.array(wavs) - np.array(target))).argmin()