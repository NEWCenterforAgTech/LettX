import os
import torch
from torchinfo import summary
import heapq
import os

class CheckpointManager:
    """
    Manages model checkpoints, keeping only the best N checkpoints based on validation loss.
    Uses a min-heap to efficiently track the best checkpoints.
    """
    def __init__(self, max_checkpoints=5, checkpoint_dir="checkpoints"):
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir = checkpoint_dir
        self.checkpoints = []  # Will be used as a min heap: (val_loss, epoch, filename)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, state, val_loss, epoch, run_name):
        """Save a checkpoint and manage the collection of best checkpoints."""
        filename = f'{run_name}_model_epoch_{epoch}.pth'
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Save the new checkpoint using safer file handling and disable new zipfile serialization
        with open(filepath, 'wb') as f:
            torch.save(state, f, _use_new_zipfile_serialization=False)
        
        # Check if this checkpoint should be tracked
        if len(self.checkpoints) < self.max_checkpoints or -val_loss > self.checkpoints[0][0]:
            # We use negative val_loss because heapq is a min-heap, but we want the best (lowest) val_losses
            checkpoint_entry = (-val_loss, epoch, filepath)
            
            if len(self.checkpoints) == self.max_checkpoints:
                # Remove the worst checkpoint
                worst = heapq.heappushpop(self.checkpoints, checkpoint_entry)
                # Delete the worst checkpoint file
                if os.path.exists(worst[2]):
                    os.remove(worst[2])
                    print(f"Removed checkpoint: {worst[2]} with val_loss {-worst[0]:.4f}")
            else:
                heapq.heappush(self.checkpoints, checkpoint_entry)
            
            print(f"Saved checkpoint: {filepath} with val_loss {val_loss:.4f}")
            return filepath
        return None
    
    def save_final_model(self, state, run_id):
        """Save the final model without affecting the checkpoint collection."""
        final_filename = f'{run_id}_final_model.pth'
        final_filepath = os.path.join(self.checkpoint_dir, final_filename)
        torch.save(state, final_filepath)
        print(f"Saved final model: {final_filepath}")
        return final_filepath
    
    def get_best_checkpoint(self):
        """Return the best checkpoint info."""
        if not self.checkpoints:
            return None
        
        best = max(self.checkpoints, key=lambda x: x[0])  # Get highest negative val_loss (lowest actual val_loss)
        return {"val_loss": -best[0], "epoch": best[1], "filepath": best[2]}