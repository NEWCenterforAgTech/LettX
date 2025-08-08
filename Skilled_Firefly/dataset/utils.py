# Description: Utility functions for dataset processing
import numpy as np
import pandas as pd

def unnormalize(predictions, gt_path):
    """
    Unnormalize predictions using min-max scaling from ground truth data.
    
    Parameters:
        predictions: Normalized predictions array
        gt_path: Path to the ground truth CSV file
        
    Returns:
        Unnormalized predictions
    """
    data = pd.read_csv(gt_path)
    
    # Convert columns to numeric, forcing errors to NaN
    for nutrient in ["N", "P", "K", "Ca", "S", "Mg"]:
        data[nutrient] = pd.to_numeric(data[nutrient], errors='coerce')

    # Drop rows with NaN values
    data = data.dropna(subset=["N", "P", "K", "Ca", "S", "Mg"])

    # Calculate the min and max for each nutrient
    mean_std_factors = {
        "N": {"mean": data["N"]},
        "P": {"mean": data["P"]},
        "K": {"mean": data["K"]},
        "Ca": {"mean": data["Ca"]},
        "S": {"mean": data["S"]},
        "Mg": {"mean": data["Mg"]}
    }
    
    unnormalized_preds = predictions.copy()
    for i, nutrient in enumerate(mean_std_factors.keys()):
        min_val = mean_std_factors[nutrient]["mean"].min()
        max_val = mean_std_factors[nutrient]["mean"].max()
        unnormalized_preds[:, i] = np.round((predictions[:, i] * (max_val - min_val)) + min_val, 2)
    return unnormalized_preds