import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import wandb
import time

# NVML for power measurement
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False
    print("[WARN] NVML not available – energy logging limited to elapsed time.")

gpus = [0] # GPU indices to measure
_gpu_handles = (
    [pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpus] if NVML_AVAILABLE else []
)

def _total_power_mw() -> int:
    """Return current total power usage (mW) across selected GPUs."""
    if not NVML_AVAILABLE:
        return 0
    return sum(pynvml.nvmlDeviceGetPowerUsage(h) for h in _gpu_handles)

# Set random seeds for reproducibility
RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Experiment configuration
EXP_NAME = "VI_trajectory_analysis_TOP_featuresT3_oneimageperday_WANDB4" 
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Set save directory
SAVE_DIR = f"/home/acohen47/changepoint/Estimated_VI_detection_different_architecture/{EXP_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Files will be saved to: {SAVE_DIR}")

wandb_dir = os.path.join(SAVE_DIR, "wandb")
os.makedirs(wandb_dir, exist_ok=True)
os.environ["WANDB_DIR"] = wandb_dir
print(f"Using wandb directory: {wandb_dir}")

# Define VI_FEATURES
VI_FEATURES = [
    'GNDVI_median', 
    'GRVI_median', 
    'NDRE_mean', 
    'NDRE_median', 
    'NDWI_median'
]

# Define window sizes to test
WINDOW_SIZES = list(range(6, 23))

# Simple Autoencoder model
class SimpleAutoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def init_wandb(feature, window_size):
    """Initialize wandb for a specific feature and window size"""
    config = {
        "feature": feature,
        "window_size": window_size,
        "random_seed": RANDOM_SEED,
        "model_type": "SimpleAutoencoder",
        "hidden_dim": 64,
        "latent_dim": 32,
        "device": str(DEVICE)
    }
    
    # Initialize wandb
    run = wandb.init(
        project="AE_Nutrient_Trajectory",
        name=f"{feature}_W{window_size}",
        config=config,
        reinit=True,
        dir=wandb_dir,
        monitor_gym=True  # Enable system monitoring
    )
    
    return run

def get_plant_trajectories(df, feature):
    """
    Extract full trajectories per plant, separated by treatment
    """
    # Group by tank_id and plant_id to get trajectories
    plant_trajectories = {}
    
    for (tank_id, plant_id), group in df.groupby(['tank_id', 'plant_id']):
        # Sort by date to ensure time order
        group = group.sort_values('date')
        
        # Extract the feature values
        if feature in group.columns:
            values = group[feature].values
            
            # Skip if contains NaN
            if np.isnan(values).any():
                continue
                
            trt = tank_id[1]  # T1, T2, T3 -> 1, 2, 3
            
            if trt not in plant_trajectories:
                plant_trajectories[trt] = []
                
            plant_trajectories[trt].append({
                'tank_id': tank_id,
                'plant_id': plant_id,
                'values': values,
                'dat_values': group['DAT'].values
            })
    
    # Print diagnostics
    for trt in plant_trajectories:
        if len(plant_trajectories[trt]) > 0:
            lengths = [len(p['values']) for p in plant_trajectories[trt]]
            print(f"Treatment {trt}: {len(plant_trajectories[trt])} plants, lengths {min(lengths)}-{max(lengths)}")
    
    return plant_trajectories

def extract_fixed_window(values, dat_values, window_size, start_dat=4):
    """
    Extract a fixed-size window from a time series starting at specified DAT
    Returns both the window values and corresponding DAT values
    """
    # Find indices where DAT >/= start_dat
    start_indices = np.where(dat_values >= start_dat)[0]
    if len(start_indices) == 0:
        return None, None  
    
    start_idx = start_indices[0]
    
    # If not enough data points after start_dat
    if start_idx + window_size > len(values):
        return None, None
    
    # Extract the window and corresponding DAT values
    window = values[start_idx:start_idx + window_size]
    window_dats = dat_values[start_idx:start_idx + window_size]
    
    return window, window_dats

def test_window_size(feature, window_size, training_plants, testing_healthy, testing_deficient):
    """
    Test a specific window size for a feature
    """
    print(f"\n==== Testing {feature} with window size {window_size} ====")
    
    run = init_wandb(feature, window_size)
    
    # Extract training data - using early time points
    train_data = []
    train_metadata = [] 
    for plant in training_plants:
        window, window_dats = extract_fixed_window(plant['values'], plant['dat_values'], window_size, start_dat=4)
        if window is not None and window_dats is not None:
            train_data.append(window)
            
            # Store metadata using DAT values
            train_metadata.append({
                'tank_id': plant['tank_id'],
                'plant_id': plant['plant_id'],
                'dat_start': window_dats[0], 
                'dat_end': window_dats[-1]    
            })
    
    healthy_windows = []
    healthy_metadata = []
    for plant in testing_healthy:
        window, window_dats = extract_fixed_window(plant['values'], plant['dat_values'], window_size, start_dat=4)
        if window is not None and window_dats is not None:
            healthy_windows.append(window)
            
            healthy_metadata.append({
                'tank_id': plant['tank_id'],
                'plant_id': plant['plant_id'],
                'dat_start': window_dats[0],
                'dat_end': window_dats[-1]
            })
    
    deficient_windows = []
    deficient_metadata = []
    for plant in testing_deficient:
        window, window_dats = extract_fixed_window(plant['values'], plant['dat_values'], window_size, start_dat=4)
        if window is not None and window_dats is not None:
            deficient_windows.append(window)
            
            deficient_metadata.append({
                'tank_id': plant['tank_id'],
                'plant_id': plant['plant_id'],
                'dat_start': window_dats[0],
                'dat_end': window_dats[-1]
            })
    
    # Make sure enough data
    if len(train_data) < 5 or len(healthy_windows) < 5 or len(deficient_windows) < 5:
        print(f"  Insufficient data for window size {window_size}")
        print(f"  Found: {len(train_data)} training, {len(healthy_windows)} healthy test, {len(deficient_windows)} deficient test")
        wandb.finish()
        return None
        
    print(f"  Training samples: {len(train_data)}")
    print(f"  Testing samples: {len(healthy_windows)} healthy, {len(deficient_windows)} deficient")
    
    # Log sample counts to wandb
    wandb.log({
        "train_samples": len(train_data),
        "healthy_test_samples": len(healthy_windows),
        "deficient_test_samples": len(deficient_windows)
    })
    
    # Convert to numpy
    train_data = np.array(train_data)
    healthy_windows = np.array(healthy_windows)
    deficient_windows = np.array(deficient_windows)
    
    # Scale the data and fit on training
    scaler = MinMaxScaler()
    scaler.fit(train_data.reshape(-1, 1))
    
    # Apply scaling
    train_data_scaled = scaler.transform(train_data.reshape(-1, 1)).reshape(train_data.shape)
    healthy_scaled = scaler.transform(healthy_windows.reshape(-1, 1)).reshape(healthy_windows.shape)
    deficient_scaled = scaler.transform(deficient_windows.reshape(-1, 1)).reshape(deficient_windows.shape)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(train_data_scaled, dtype=torch.float32)
    X_healthy = torch.tensor(healthy_scaled, dtype=torch.float32)
    X_deficient = torch.tensor(deficient_scaled, dtype=torch.float32)
    
    # Training dataset and loader
    train_dataset = TensorDataset(X_train)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Model and optimizer setup
    model = SimpleAutoencoder(window_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
  
    wandb.watch(model, log="all", log_freq=100000)
    
    # Record start time
    train_start_t = time.time()
    train_start_p = _total_power_mw()
    
    # Training loop
    print("  Training autoencoder...")
    training_losses = []
    
    for epoch in range(100):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = batch[0].to(DEVICE)
         
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
         
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average loss for this epoch
        epoch_loss = total_loss / len(train_loader)
        training_losses.append(epoch_loss)
        
        wandb.log({"epoch": epoch, "train_loss": epoch_loss})
        
        # Print every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/100, Loss: {epoch_loss:.6f}")
    
    train_end_t = time.time()
    train_end_p = _total_power_mw()

    train_elapsed_h = (train_end_t - train_start_t) / 3600
    train_avg_w = (train_start_p + train_end_p) / 2 / 1000  # mW→W
    train_energy_wh = train_avg_w * train_elapsed_h

    print("\n=== Training Energy & Time ===")
    print(f"Elapsed: {train_elapsed_h:.4f} h | Avg Power: {train_avg_w:.2f} W | "
      f"Energy: {train_energy_wh:.4f} Wh")

    wandb.log({"train_elapsed_h": train_elapsed_h,
           "train_avg_power_w": train_avg_w,
           "train_energy_wh": train_energy_wh})

    final_train_loss = total_loss / len(train_loader)
        
    # Evaluate on test sets
    # Record start time for inference energy measurement
    inf_start_t = time.time()
    inf_start_p = _total_power_mw()

    print("  Evaluating...")
    model.eval()
    with torch.no_grad():
        # Calculate reconstruction errors for healthy windows
        healthy_errors = []
        for x in X_healthy:
            x = x.to(DEVICE)
            recon = model(x)
            error = torch.mean((recon - x) ** 2).item()
            healthy_errors.append(error)
        
        # Calculate reconstruction errors for deficient windows
        deficient_errors = []
        for x in X_deficient:
            x = x.to(DEVICE)
            recon = model(x)
            error = torch.mean((recon - x) ** 2).item()
            deficient_errors.append(error)
    
    # Calculate threshold using 1.5 * final training loss
    threshold = 1.5 * final_train_loss
    
    # Calculate detection rates
    fpr = sum(e > threshold for e in healthy_errors) / len(healthy_errors)
    tpr = sum(e > threshold for e in deficient_errors) / len(deficient_errors)
    net_rate = tpr - fpr

    # Record end time and calculate testing duration
    inf_end_t = time.time()
    inf_end_p = _total_power_mw()

    inf_elapsed_h = (inf_end_t - inf_start_t) / 3600
    inf_avg_w = (inf_start_p + inf_end_p) / 2 / 1000
    inf_energy_wh = inf_avg_w * inf_elapsed_h

    print("\n=== Inference Energy & Time ===")
    print(f"Elapsed: {inf_elapsed_h:.4f} h | Avg Power: {inf_avg_w:.2f} W | "
      f"Energy: {inf_energy_wh:.4f} Wh")

    wandb.log({"inference_elapsed_h": inf_elapsed_h,
           "inference_avg_power_w": inf_avg_w,
           "inference_energy_wh": inf_energy_wh})

    
    # Wait for wandb
    time.sleep(1)
    
    
    # Log metrics to wandb
    wandb.log({
        "threshold": threshold,
        "false_positive_rate": fpr,
        "true_positive_rate": tpr,
        "net_detection_rate": net_rate,
        "final_train_loss": final_train_loss
    })
    
    # Print results
    print(f"  Results for {feature} window size {window_size}:")
    print(f"    Threshold: {threshold:.6f}")
    print(f"    False Positive Rate: {fpr:.2%}")
    print(f"    True Positive Rate: {tpr:.2%}")
    print(f"    Net Detection Rate: {net_rate:.2%}")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(healthy_errors, bins=15, alpha=0.7, color='green', label='Normal (100%)')
    plt.hist(deficient_errors, bins=15, alpha=0.7, color='red', label='Severely Deficient (25%)')
    plt.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title(f'{feature} - Window Size {window_size}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save histogram
    clean_name = feature.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(os.path.join(SAVE_DIR, 'plots', f'{clean_name}_W{window_size}_histogram.png'))
    
    wandb.log({"reconstruction_error_histogram": wandb.Image(plt)})
    plt.close()
    
    # Plot training loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_losses) + 1), training_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss - {feature} W{window_size}')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(SAVE_DIR, 'plots', f'{clean_name}_W{window_size}_loss.png'))
    
    wandb.log({"training_loss_curve": wandb.Image(plt)})
    plt.close()
    
    # Save results + metadata for further analysis
    healthy_results = pd.DataFrame(healthy_metadata)
    healthy_results['error'] = healthy_errors
    healthy_results['above_threshold'] = [e > threshold for e in healthy_errors]
    healthy_results.to_csv(os.path.join(SAVE_DIR, 'data', 
                                      f'{clean_name}_W{window_size}_healthy_errors.csv'), index=False)
    
    deficient_results = pd.DataFrame(deficient_metadata)
    deficient_results['error'] = deficient_errors
    deficient_results['above_threshold'] = [e > threshold for e in deficient_errors]
    deficient_results.to_csv(os.path.join(SAVE_DIR, 'data', 
                                        f'{clean_name}_W{window_size}_deficient_errors.csv'), index=False)
    
    wandb.finish()
    
    # Return results
    return {
        "feature": feature,
        "window_size": window_size,
        "TPR": tpr,
        "FPR": fpr,
        "net_rate": net_rate,
        "threshold": threshold,
        "healthy_count": len(healthy_errors),
        "deficient_count": len(deficient_errors),
   
    }

def analyze_feature(feature, df):
    """
    Analyze a feature across multiple window sizes
    """
    print(f"\n{'='*20} Analyzing {feature} {'='*20}")
    
    plant_trajectories = get_plant_trajectories(df, feature)
    
    # Check if both healthy (T1) and deficient (T3) plants
    if '1' not in plant_trajectories or '3' not in plant_trajectories:
        print(f"Missing data for {feature}. Need both T1 and T3 plants.")
        return None
    
    # Randomize order of T1 plants, use first 40 T1 plants for training (or as many as available), and remaining for healthy testing
    random.seed(RANDOM_SEED)
    random.shuffle(plant_trajectories['1'])
    
    training_plants = plant_trajectories['1'][:min(40, len(plant_trajectories['1']))]
    testing_healthy = plant_trajectories['1'][min(40, len(plant_trajectories['1'])):]
    
    # Use all T3 plants for testing 'deficient' trajectories
    testing_deficient = plant_trajectories['3']
    
    print(f"Training set: {len(training_plants)} T1 plants")
    print(f"Testing sets: {len(testing_healthy)} T1 plants and {len(testing_deficient)} T3 plants")
    
    # Results for all window sizes
    window_results = []
    
    for window_size in WINDOW_SIZES:
        result = test_window_size(feature, window_size, training_plants, testing_healthy, testing_deficient)
        if result:
            window_results.append(result)
    
    # Save results
    if window_results:
        results_df = pd.DataFrame(window_results)
        clean_name = feature.replace(" ", "_").replace("(", "").replace(")", "")
        results_df.to_csv(os.path.join(SAVE_DIR, 'results', f'{clean_name}_results.csv'), index=False)
        
        # Plot by window size
        plt.figure(figsize=(12, 6))
        plt.plot(results_df["window_size"], results_df["TPR"], 'g-', marker='o', label="TPR")
        plt.plot(results_df["window_size"], results_df["FPR"], 'r-', marker='x', label="FPR")
        plt.plot(results_df["window_size"], results_df["net_rate"], 'b-', marker='s', label="Net Rate")
        plt.xlabel("Window Size (Days)")
        plt.ylabel("Rate")
        plt.title(f"Detection Performance by Window Size - {feature}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'plots', f'{clean_name}_by_window.png'))
        plt.close()
        
        # Find best window size
        best_result = results_df.loc[results_df['net_rate'].idxmax()]
        print(f"\nBest window size for {feature}: {best_result['window_size']}")
        print(f"  TPR: {best_result['TPR']:.2%}, FPR: {best_result['FPR']:.2%}, Net Rate: {best_result['net_rate']:.2%}")
        
        return results_df
    else:
        print(f"No valid results for {feature}")
        return None

# Model
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df = pd.read_csv("/home/acohen47/Data_GT_and_Estimated_Trajectories/vi_MS_results_All_channels_included.csv")
    df.dropna(inplace=True)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.sort_values(by=['date', 'tank_id', 'plant_id'])

    # Create DAT column based on first date
    transplant_date = pd.to_datetime('10/15/24')
    print(f"Using {transplant_date} as transplant date (DAT = 0)")
    df['DAT'] = (df['date'] - transplant_date).dt.days
    print(f"Calculated DAT range: {df['DAT'].min()} to {df['DAT'].max()}")

    # Check if all features exist in the dataframe
    available_features = [f for f in VI_FEATURES if f in df.columns]
    if len(available_features) < len(VI_FEATURES):
        missing = set(VI_FEATURES) - set(available_features)
        print(f"Warning: The following features are missing: {missing}")
        print(f"Proceeding with available features: {available_features}")
        VI_FEATURES = available_features

    print(f"Analyzing {len(VI_FEATURES)} features with window sizes {WINDOW_SIZES}")

    # Process all features
    all_results = []
    for feature in VI_FEATURES:
        results = analyze_feature(feature, df)
        if results is not None:
            all_results.append(results)

    # Combine results
    if all_results:
        combined_results = pd.concat(all_results)
        combined_results.to_csv(os.path.join(SAVE_DIR, 'results', 'all_results.csv'), index=False)
        
        # Find best window size for each feature
        best_by_feature = combined_results.loc[combined_results.groupby('feature')['net_rate'].idxmax()]
        best_by_feature = best_by_feature.sort_values('net_rate', ascending=False)
        best_by_feature.to_csv(os.path.join(SAVE_DIR, 'results', 'best_combinations.csv'), index=False)
        
        # Create heatmap
        pivot_net = combined_results.pivot_table(values="net_rate", index="feature", columns="window_size")
        plt.figure(figsize=(15, 8))
        sns.heatmap(pivot_net, annot=True, fmt=".2f", cmap="RdYlGn", 
                   vmin=-0.2, vmax=0.8, linewidths=.5, cbar_kws={'label': 'Net Detection Rate'})
        plt.title('Net Detection Rate by Feature and Window Size', fontsize=16)
        plt.xlabel('Window Size (Days)', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'plots', 'feature_window_heatmap.png'), dpi=300)
        plt.close()
        
        # Create true and false heatmaps
        for metric, title, cmap in [
            ("TPR", "True Positive Rate", "YlGnBu"),
            ("FPR", "False Positive Rate", "YlOrRd")
        ]:
            pivot = combined_results.pivot_table(values=metric, index="feature", columns="window_size")
            plt.figure(figsize=(15, 8))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, 
                       vmin=0, vmax=1, linewidths=.5, cbar_kws={'label': title})
            plt.title(f'{title} by Feature and Window Size', fontsize=16)
            plt.xlabel('Window Size (Days)', fontsize=14)
            plt.ylabel('Feature', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, 'plots', f'{metric.lower()}_heatmap.png'), dpi=300)
            plt.close()
        
        # Summary bar chart
        plt.figure(figsize=(14, 8))
        bars = plt.barh(best_by_feature['feature'], best_by_feature['net_rate'], color='skyblue')
        plt.xlabel('Net Detection Performance (TPR-FPR)')
        plt.ylabel('Feature')
        plt.title('Best Window Sizes for Nutrient Deficiency Detection')
        plt.grid(alpha=0.3)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            window_label = best_by_feature.iloc[i]['window_size']
            plt.text(max(0.01, width + 0.02), bar.get_y() + bar.get_height()/2, 
                    f'{best_by_feature.iloc[i]["net_rate"]:.2%} (W{window_label})', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'plots', 'best_feature_summary.png'))
        plt.close()
        
        # Print summary
        print("\n===== BEST FEATURE-WINDOW COMBINATIONS =====")
        print("{:<15} {:<12} {:<10} {:<10} {:<10}".format("Feature", "Window Size", "TPR", "FPR", "Net Rate"))
        print("-" * 65)
        for _, row in best_by_feature.iterrows():
            print("{:<15} {:<12} {:<10.2%} {:<10.2%} {:<10.2%}".format(
                row['feature'], row['window_size'], 
                row['TPR'], row['FPR'], row['net_rate']))

    print("\nAnalysis complete!")