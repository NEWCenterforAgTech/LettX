import sys
import os
# Add the parent directory to sys.path to enable absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re

import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from skimage.transform import resize


class LettxDataset(Dataset):
    """Dataset for GT+UGA hypercubes and csv ground truths."""
    # Note: The UGA dataset is mounted at /data/lettx

    def __init__(self, csv_file, root_dir, img_size=(512, 512), transforms=[], device='cuda:1'):
        """
        Arguments:
            csv_file (string): Path to the csv file with ground truths concentrations.
            root_dir (string): Path to directory containing the hypercubes (i.e. the preprocessed .npy files).
            img_size (tuple): Desired image size (height, width).
            transform (list, optional): List of transform function to be applied to hypercube.
                All transforms should be applicable to (h,w,c) np array and should be callable with no other
                argument than the hypercube itself.
            device (string): Device to use for tensor operations.
        """
        self.gt_data = pd.read_excel(csv_file, sheet_name=None)
        self.root_dir = root_dir
        # self.fnames = [x for x in os.listdir(root_dir) if x.endswith(".npy")]
        # Load all .npy files
        all_files = [x for x in os.listdir(root_dir) if x.endswith(".npy")]
        invalid_files_path = "/home/vmuriki3/deeplearning_nutrient_estimation/model/invalid_arr.txt"
        # Filter out invalid files if a list is provided
        if invalid_files_path and os.path.exists(invalid_files_path):
            invalid_files = []
            with open(invalid_files_path, 'r') as f:
                for line in f:
                    # Clean up the line and extract just the base filename
                    filepath = line.strip().strip("'").strip(",").strip()
                    if filepath:
                        # Extract only the basename part (filename without directory)
                        base_name = os.path.basename(filepath).split('_')
                        base_name = '_'.join(base_name[:4]) + '.npy'
                        # Replace rgb.jpg with npy - clean up any extra quotes/apostrophes
                        invalid_files.append(base_name)
            
            print(f"Loaded {len(invalid_files)} invalid files to exclude")
            
            # Add a debug print to see a few examples
            print(f"Sample invalid files (first 5): {invalid_files[:5]}")
            
            # Filter out invalid files with more careful matching
            self.fnames = []
            for file in all_files:
                # Clean both filenames for comparison
                clean_file = file.strip("'")
                if clean_file not in [f.strip("'") for f in invalid_files] and file[0] in ['1', '2', '3', '4', '5', '6', '7']:
                    self.fnames.append(file)
            
            print(f"Kept {len(self.fnames)} out of {len(all_files)} files")
            # print(f"Sample valid files (first 5): {self.fnames}")
        else:
            self.fnames = all_files
        self.img_size = img_size
        self.transforms = transforms
        self.nutrient_labels = ['N', 'P', 'K', 'Ca', 'Mg', 'S']
        self.device = device
        self.cache = {}
        data = np.load("/home/acohen47/HSI_ViT/deeplearning_nutrient_estimation/dataset/final_dataset_stats1.npy", allow_pickle=True).item()['total']
        self.mean, self.std = data['mean'], data['std']
        
        self.normalized_gt_data = {}
        for sheet_name, sheet_data in self.gt_data.items():
            self.normalized_gt_data[sheet_name] = self.get_normalized_data(sheet_data)
            
        self.white, self.dark = load_white_black_img()

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image, label, filename) where image is the hypercube data,
                  label is the ground truth nutrient values, and filename is the source file name
        """
        # Read image
        fname = self.fnames[idx]
        fpath = os.path.join(self.root_dir, fname)

        try:
            img = np.load(fpath, mmap_mode='r')
            # ? See a more efficient way to do this
            if img.shape[0] != self.img_size[0] or img.shape[1] != self.img_size[0]:
                img = resize(img, (self.img_size[0], self.img_size[1], img.shape[2]), preserve_range=True).astype(np.float32)
                print("Resized image to 512x512")
            eps = 1e-8
            # self.mean is of type float32
            img = (img - self.mean) / (self.std + eps)
        except Exception as e:
            print(f"Error loading {fpath}: {str(e)}")
            return None
        
        # if img.shape != (1024, 1280, 141) or img.mean() == 1.0:
        #     print(f"Invalid shape {img.shape} for {fpath}")
        #     return None
        
        default_label = {nutrient: 0.5 for nutrient in self.nutrient_labels}
        label = default_label
        
        # Get label from filename
        try:
            label = self.get_gt_data_from_fname(fname, self.normalized_gt_data)
            if label is None:
                print(f"No label found for {fname}")
                label = default_label
        except Exception as e:
            print(f"Error extracting label for {fname}: {str(e)}, label={label}")
            label = default_label

        # Apply transformations if any
        # if self.transforms:
        #     for t in self.transforms:
        #         img = t(img)
        
        end_time1 = time.time()
        # print(f"Time taken before {fname}: {end_time1 - end_time}")
        img = torch.from_numpy(img).to(dtype=torch.float16)
        label = torch.tensor(list(label.values()), dtype=torch.float16)
        end_time2 = time.time()
        print(f"Time taken totally {fname}: {end_time2 - end_time1}")
        return img, label, fname
    
    def get_gt_data_from_fname(self, fname, gt_data):
        """
        Extract ground truth data from the filename.
        
        Args:
            fname (str): Filename to parse
            gt_data (DataFrame): Normalized ground truth data
            
        Returns:
            dict: Dictionary of nutrient values
        """
        # print(f"Extracting ground truth data for {fname}")
        file_name = fname[:8]
        experiment_id = 'H' + fname[0]
        gt_data = gt_data[experiment_id]
        filtered_data = gt_data[gt_data['File Name'].str.contains(file_name)]
        if filtered_data.empty:
            print(f"No matching ground truth data for {fname} with File Name={file_name}")
            return None

        # Extract nutrient values
        label = {}
        for nutrient in self.nutrient_labels:
            try:
                val = filtered_data[nutrient].iloc[0]
                
                # Handle missing values (represented as '.')
                if val == '.':
                    val = gt_data[nutrient].mean()
                    print(f"Missing value for {nutrient} in {fname}, using mean: {val}")
                
                val = float(val)
                
                # Check for negative values which might indicate data issues
                if val < 0:
                    print(f"Warning: Negative value {val} found for {nutrient} in {fname}")
                
                label[nutrient] = val
            except (ValueError, IndexError) as e:
                print(f"Error extracting {nutrient} value for {fname}: {str(e)}")
                label[nutrient] = gt_data[nutrient].mean()
        # print(f"Extracted label: {label} for {fname}")
        return label
    
    def get_normalized_data(self, data):
        # data is a dictionary of DataFrames (multiple harvests)
        
        # Convert columns to numeric, forcing errors to NaN (coerce)
        data["N"] = pd.to_numeric(data["N"], errors='coerce')
        data["P"] = pd.to_numeric(data["P"], errors='coerce')
        data["K"] = pd.to_numeric(data["K"], errors='coerce')
        data["Ca"] = pd.to_numeric(data["Ca"], errors='coerce')
        data["S"] = pd.to_numeric(data["S"], errors='coerce')
        data["Mg"] = pd.to_numeric(data["Mg"], errors='coerce')

        # Drop rows with NaN values
        data = data.dropna(subset=["N", "P", "K", "Ca", "S", "Mg"])

        mean_std_factors = {
            "N": {"values": data["N"]},
            "P": {"values": data["P"]},
            "K": {"values": data["K"]},
            "Ca": {"values": data["Ca"]},
            "S": {"values": data["S"]},
            "Mg": {"values": data["Mg"]}
        }
        
        normalized_data = data.copy()
        for nutrient in mean_std_factors.keys():
            # Min-max normalization between 0 and 1
            min_value = mean_std_factors[nutrient]["values"].min()
            max_value = mean_std_factors[nutrient]["values"].max()
            normalized_data[nutrient] = (mean_std_factors[nutrient]["values"] - min_value) / (max_value - min_value)

        return normalized_data

def lettx_collate_fn(batch):
    """
    Custom collate function to handle the `None` labels
    when reading CSV files with potentially missing data.
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if not batch:
        # raise ValueError("Batch is empty after filtering None values")
        return None
    
    # Unpack batch
    images, ground_truths, fnames = zip(*batch)
    
    # Stack and format images
    # PyTorch's neural network modules expect input images in the format (Batch, Channels, Height, Width)
    images = torch.stack([img.permute(2, 0, 1).float() if img.dim() == 3 else img for img in images])
    ground_truths = torch.stack(ground_truths)

    return images, ground_truths, fnames

def load_white_black_img(dark_pth=None, white_pth=None):
    
    dark_pth = "/mnt/lettx/hsi_processed_data/processed_512/Dark.npy"
    dark = np.load(dark_pth, mmap_mode='r')

    white_pth = "/mnt/lettx/hsi_processed_data/calibration_512/calibration_1212_1141.npy"
    white = np.load(white_pth, mmap_mode='r')

    start_col = int(white.shape[1] * 0.3)

    # Crop the white image
    cropped_white = white[:, start_col:, :]

    # Resize the cropped image to 512x512
    resized_white = resize(
        cropped_white,
        (512, 512, cropped_white.shape[2]),
        preserve_range=True,
        anti_aliasing=True
    ).astype(np.float16)
    
    return resized_white, dark