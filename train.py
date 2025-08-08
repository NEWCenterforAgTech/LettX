import sys
import os
# Add the parent directory to sys.path to enable absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import r2_score, mean_squared_error
from dataset import LettxDataset, lettx_collate_fn, hs_train_transforms, hs_val_transforms, hs_test_transforms
from model import ViTModel, HSResNet34, HSResNet18, PretrainedViTModel, ViTModelSmall, ViTModelCCT, HSResNet50
from utils import CheckpointManager
import argparse
import numpy as np
import wandb
from torchinfo import summary
import os
from tqdm import tqdm
import time
import sys
import shutil


# Define default hyperparameters
def get_default_config(model_name="ViTModel"):
    return {
        "num_epochs": 100,
        "learning_rate": 1e-5,
        "batch_size": 16,
        "input_channels": 141,
        "output_channels": 141,
        "model": model_name,
        "num_outputs": 6,
        "img_size": (512, 512),
        "crop_size": (224, 224),
        "checkpoint_dir": "checkpoints",
        "wandb_project": "Abi LettX",
        "wandb_entity": "abigailrcohen-georgia-institute-of-technology",
        "num_workers": 1,
        "device": "cuda:1" if torch.cuda.is_available() else "cpu",
        "max_checkpoints": 3  # Keep only the best 5 checkpoints
    }


# Define save checkpoint function - keeping for backward compatibility
def save_checkpoint(state, filename, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)

def initialize_weights(m):
    # Initialize weights using Xavier initialization for only head layers (last layers)
    if hasattr(m, 'head'):
        for m in m.head.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    # # Initialize weights for channel projector if it exists
    # if hasattr(m, 'channel_projector'):
    #     for m in m.channel_projector.modules():
    #         if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)


def calculate_metrics(targets, preds, prefix=""):
    """Calculate and return metrics in a dictionary format."""
    metrics = {}
    
    # Overall metrics
    overall_r2 = r2_score(targets, preds)
    overall_rmse = np.sqrt(mean_squared_error(targets, preds))
    metrics[f"{prefix}_r2"] = overall_r2
    metrics[f"{prefix}_rmse"] = overall_rmse
    
    # Per-nutrient metrics
    r2_scores = r2_score(targets, preds, multioutput='raw_values')
    cols = ["N", "P", "K", "Ca", "Mg", "S"][:len(r2_scores)]
    if not isinstance(r2_scores, float):
        for i, col in enumerate(cols):
            metrics[f"{prefix}_{col}_r2"] = r2_scores[i]
    print("metrics:", metrics)
    return metrics

def train(model, train_loader, optimizer, epoch, device, criterion):
    model.train()
    train_losses, train_preds, train_targets = [], [], []
    
    # Create progress bar for training
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False)
    
    for inputs, targets, _ in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        targets = torch.clamp(targets.to(device, dtype=torch.float, non_blocking=True), min=0)
        
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_losses.append(loss.item())
        train_preds.append(outputs.detach().cpu().numpy())
        train_targets.append(targets.cpu().numpy())
        
        # Update progress bar with current loss
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", refresh=True)
        
        # Clear GPU memory
        del inputs, targets, outputs, loss
    
    torch.cuda.empty_cache()
    
    # Calculate metrics
    train_preds = np.concatenate(train_preds)
    train_targets = np.concatenate(train_targets)
    metrics = calculate_metrics(train_targets, train_preds, prefix="train")
    metrics["train_loss"] = np.mean(train_losses)
    metrics["epoch"] = epoch
    
    return np.mean(train_losses), metrics

def validate(model, val_loader, epoch, device, criterion):
    model.eval()
    val_losses, val_preds, val_targets = [], [], []

    # Create progress bar for validation
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False)
    
    with torch.no_grad():
        for inputs, targets, _ in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            targets = torch.clamp(targets.to(device, dtype=torch.float, non_blocking=True), min=0)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_losses.append(loss.item())
            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
            
            # Update progress bar with current loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", refresh=True)
            
            # Clear GPU memory
            del inputs, targets, outputs, loss
    
    torch.cuda.empty_cache()
    
    # Calculate metrics
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    metrics = calculate_metrics(val_targets, val_preds, prefix="val")
    metrics["val_loss"] = np.mean(val_losses)
    metrics["epoch"] = epoch
    
    return np.mean(val_losses), metrics

def test(model, test_loader, device, criterion):
    """
    Evaluate the model on the test dataset.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for the test dataset
        device: Device to run the evaluation on
        criterion: Loss function
        
    Returns:
        tuple: (mean_loss, metrics_dictionary)
    """
    model.eval()
    test_losses, test_preds, test_targets = [], [], []

    # Create progress bar for test evaluation
    progress_bar = tqdm(test_loader, desc="Test Evaluation", leave=False)
    
    with torch.no_grad():
        for inputs, targets, _ in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            targets = torch.clamp(targets.to(device, dtype=torch.float, non_blocking=True), min=0)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_losses.append(loss.item())
            test_preds.append(outputs.cpu().numpy())
            test_targets.append(targets.cpu().numpy())
            
            # Update progress bar with current loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", refresh=True)
            
            # Clear GPU memory
            del inputs, targets, outputs, loss
    
    torch.device.empty_cache()
    
    # Calculate metrics
    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)
    metrics = calculate_metrics(test_targets, test_preds, prefix="test")
    metrics["test_loss"] = np.mean(test_losses)
    
    print("\n===== TEST RESULTS =====")
    print(f"Test Loss: {np.mean(test_losses):.4f}")
    print(f"Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")
    print("========================\n")
    
    return np.mean(test_losses), metrics

def create_model(config, device):
    """Create and initialize the model based on configuration."""

    model_name = config["model"]
    
    if model_name == "ViTModel":
        model = ViTModel(
            image_size=config["img_size"][0],
            input_channels=config["input_channels"], 
            num_outputs=config["num_outputs"],
            output_channels=config["output_channels"],
        ).to(device)

    elif model_name == "HSResNet34":
        model = HSResNet34(
            input_channels=config["input_channels"],
            num_outputs=config["num_outputs"],
            output_channels=config["output_channels"]
        ).to(device)

    elif model_name == "HSResNet50":
        model = HSResNet50(
            input_channels=config["input_channels"],
            num_outputs=config["num_outputs"],
            output_channels=config["output_channels"]
        ).to(device)
    
    elif model_name == "HSResNet18":
        model = HSResNet18(
            input_channels=config["input_channels"],
            num_outputs=config["num_outputs"],
            output_channels=config["output_channels"]
        ).to(device)
    
    elif model_name == "PretrainedViTModel":
        model = PretrainedViTModel(
            input_channels=config["input_channels"],
            num_outputs=config["num_outputs"],
            output_channels=config["output_channels"]
        ).to(device)
    
    elif model_name == "ViTModelSmall":
        model = ViTModelSmall(
            input_channels=config["input_channels"],
            num_outputs=config["num_outputs"],
            output_channels=config["output_channels"]
        ).to(device)
    
        # Uncomment and modify the following lines if using a CCT model
    
    # Instantiate the model for images with shape (512, 512, 141)
    # ! Check the CCT papar for additional dimensions
    # model = ViTModelCCT(
    #         input_channels=config["input_channels"],
    #         num_outputs=config["num_outputs"],
    #         output_channels=3,
    #         image_size=config["img_size"][0],
    #         patch_size=16,       # patch_size can be used if required by your CCT module design.
    #         ).to(device)  # We're not altering channels here.
    
    # # Make it all trainable
    for param in model.parameters():
        param.requires_grad = True

    print("Parameter trainable status:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")

    #model.apply(initialize_weights)
    return model

# def save_code_to_wandb(project_root, wandb_run_name):
    """
    Save all Python files in the project to W&B for experiment tracking.
    
    Args:
        project_root (str): Root directory of the project
    """
    print("Saving code to W&B...")
    
    # Create a list of Python files to save
    python_files = [
        '/home/vmuriki3/deeplearning_nutrient_estimation/train/train.py',
        '/home/vmuriki3/deeplearning_nutrient_estimation/train/utils.py'
    ]
    
    # Add Python files from specific folders
    specific_folders = ["dataset", "model"]
    for folder in specific_folders:
        folder_path = os.path.join(project_root, folder)
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.py'):
                    full_path = os.path.join(folder_path, file)
                    python_files.append(full_path)
                
    # Save each file to W&B
    for file_path in python_files:
        rel_path = os.path.relpath(file_path, project_root)
        wandb.save(file_path, base_path=project_root)
        print(f"Saved {rel_path} to W&B")
    
    print(f"Saved {len(python_files)} Python files to W&B")
    
    def save_files_locally(python_files, project_root, wandb_run_name):
        """
        Save all tracked Python files to a local folder for the current run.
        
        Args:
            python_files (list): List of files to save
            project_root (str): Root directory of the project
            wandb_run_name (str): Name of the current W&B run
            
        Returns:
            str: Path to the saved files directory
        """
        
        # Create a directory for saved runs
        saved_runs_dir = os.path.join(project_root, "saved_runs")
        os.makedirs(saved_runs_dir, exist_ok=True)
        
        # Create a directory for this specific run
        run_dir = os.path.join(saved_runs_dir, wandb_run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"Saving code files locally to {run_dir}...")
        
        # Save each file to the local directory
        for file_path in python_files:
            rel_path = os.path.relpath(file_path, project_root)
            target_path = os.path.join(run_dir, rel_path)
            
            # Create the necessary directories
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Copy the file
            shutil.copy2(file_path, target_path)
            print(f"Saved {rel_path} locally")
        
        print(f"Saved {len(python_files)} Python files locally to {run_dir}")
        return run_dir

    # Save files locally as well
    local_save_path = save_files_locally(python_files, project_root, wandb_run_name)
    print(f"Code saved to W&B and locally at {local_save_path}")

def main():
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Train a model for nutrient estimation")
    parser.add_argument(
        "--root", "-r", required=True, type=str,
        help="Path to folder containing the data",
    )
    parser.add_argument(
        "--groundTruth", "-gt", required=True, type=str,
        help="Path to folder containing the GT data",
    )
    parser.add_argument(
        "--modelName", "-m", required=True, type=str,
        help="Name of the model to use (e.g., ViTModel, ViTCCT, HSResNet18, etc.)",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Number of epochs")

    args = parser.parse_args()
    
    # Get project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Load default configuration and update with command line args
    config = get_default_config(model_name=args.modelName)
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["learning_rate"] = args.lr
    if args.epochs:
        config["num_epochs"] = args.epochs

    device = torch.device(config["device"])
    criterion = nn.MSELoss()
    # Initialize dataset
    full_dataset = LettxDataset(csv_file=args.groundTruth, root_dir=args.root, img_size=config["img_size"], transforms=None, device=device)

    # Split the dataset
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Croping the images
    train_dataset.dataset.transforms = hs_train_transforms(crop_size=config["img_size"])
    val_dataset.dataset.transforms = hs_val_transforms(crop_size=config["img_size"])
    test_dataset.dataset.transforms = hs_test_transforms(crop_size=config["img_size"])

    # Create optimized DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,  # Shuffle training data
        collate_fn=lettx_collate_fn,
        drop_last=True,
        num_workers=config["num_workers"],
        persistent_workers=True,
        # pin_memory=True,  # Speed up data transfer to GPU
        # prefetch_factor=2  # Prefetch batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=lettx_collate_fn,
        drop_last=False,
        num_workers=config["num_workers"],
        persistent_workers=True,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=lettx_collate_fn,
        drop_last=False,
        num_workers=config["num_workers"],
        persistent_workers=True,
        pin_memory=True,
    )
    # Print dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Initialize the model
    model = create_model(config, device)
    
    # Get run notes from user
    print("\n" + "="*50)
    run_notes = input("Enter notes for this run (will be saved to wandb): ")
    print("="*50 + "\n")
    
    # Initialize wandb
    wandb.init(
        project=config["wandb_project"],
        entity=config["wandb_entity"],
        config=config,
        reinit=True,
        notes=run_notes
    )
    wandb.watch(model, log="all", log_freq=10)
    
    # Save all Python code files to W&B
   #  save_code_to_wandb(project_root, wandb.run.name)
    
    print("Model initialized: ", wandb.run.name)
    
    print("\n===== DETAILED MODEL SUMMARY =====")
    summary(model, 
            input_size=(config["batch_size"], config["input_channels"], 
                      config["img_size"][0], config["img_size"][1]),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            depth=4)
    print("==================================\n")
    
    # Initialize the checkpoint manager
    checkpoint_manager = CheckpointManager(
        max_checkpoints=config["max_checkpoints"], 
        checkpoint_dir=config["checkpoint_dir"]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=1e-5)
    # Dynamic learning rate scheduler - reduces by 10x (1e-6->1e-7) if validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3, 
    )

    best_val_loss = float('inf')
    
    # Training loop
    epoch_progress = tqdm(range(config["num_epochs"]), desc="Training Progress", position=0)
    
    for epoch in epoch_progress:
        train_loss, train_metrics = train(model, train_loader, optimizer, epoch, device, criterion)
        val_loss, val_metrics = validate(model, val_loader, epoch, device, criterion)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Log all metrics to wandb
        log_metrics = {**train_metrics, **val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        wandb.log(log_metrics)

        # Update epoch progress bar with summary metrics
        epoch_progress.set_postfix({
            'train_loss': f"{train_loss:.4f}", 
            'val_loss': f"{val_loss:.4f}",
            'train_rmse': f"{train_metrics['train_rmse']:.4f}",
            'val_rmse': f"{val_metrics['val_rmse']:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })
        
        # Print more detailed metrics after each epoch
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, RMSE: {train_metrics['train_rmse']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, RMSE: {val_metrics['val_rmse']:.4f}")
        
        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': config
            }
            
            # Use the checkpoint manager to save and manage checkpoints
            checkpoint_manager.save_checkpoint(
                checkpoint, 
                val_loss,
                epoch,
                wandb.run.name
            )
            
            wandb.run.summary["best_val_loss"] = best_val_loss
            epoch_progress.set_postfix_str(epoch_progress.postfix + " [saved]")
    
    # Save final model
    final_checkpoint = {
        'epoch': config["num_epochs"] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }
    
    # test the mode
    
    test_loss, test_metrics = test(model, test_loader, device, criterion)
    wandb.log(test_metrics)
    
    checkpoint_manager.save_final_model(final_checkpoint, wandb.run.name)
    
    # Log the best checkpoint info to wandb
    best_checkpoint = checkpoint_manager.get_best_checkpoint()
    if best_checkpoint:
        wandb.run.summary.update({
            "best_checkpoint_epoch": best_checkpoint["epoch"],
            "best_checkpoint_val_loss": best_checkpoint["val_loss"],
            "best_checkpoint_path": best_checkpoint["filepath"]
        })

    print("Training completed.")
    wandb.finish()

if __name__ == "__main__":
    main()