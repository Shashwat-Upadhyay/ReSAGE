import os
import torch
from torch.utils.data import Dataset
import numpy as np

class ReSAGEDataset(Dataset):
    """
    ReSAGE PyTorch Dataset for lazy loading genomic/microscopy npy arrays.
    Only loads data from disk when __getitem__ is called, preventing RAM bloat.
    """
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Base path to the ReSAGE data directory (e.g., 'd:/My_Projects/ReSAGE/data')
            split (str): Split to load: 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.split_dir = os.path.join(root_dir, split)
        self.noisy_dir = os.path.join(self.split_dir, "NoisyLR")
        self.gt_dir = os.path.join(self.split_dir, "GT")
        
        if not os.path.exists(self.noisy_dir):
            raise FileNotFoundError(f"NoisyLR directory not found at: {self.noisy_dir}")
            
        # Get sorted filenames to ensure consistent pairing
        self.filenames = sorted([f for f in os.listdir(self.noisy_dir) if f.endswith(".npy")])
        
        # Check if GT directory exists (typically only for train/val splits)
        self.has_gt = os.path.exists(self.gt_dir)
        if self.has_gt:
            # Verify GT directory files match NoisyLR files
            gt_files = sorted([f for f in os.listdir(self.gt_dir) if f.endswith(".npy")])
            if len(self.filenames) != len(gt_files):
                raise ValueError(
                    f"Mismatch in number of files: NoisyLR has {len(self.filenames)}, GT has {len(gt_files)}"
                )
            # Optional warning if filenames don't match exactly
            if self.filenames != gt_files:
                print(f"Warning: Filenames in NoisyLR and GT splits do not match exactly.")
                
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # Lazy load Noisy Low-Resolution image
        noisy_path = os.path.join(self.noisy_dir, filename)
        noisy_np = np.load(noisy_path).astype(np.float32)
        
        # Sanity check input shape
        if noisy_np.shape != (128, 128):
            raise ValueError(f"Expected NoisyLR shape (128, 128), but got {noisy_np.shape} for {filename}")
            
        # Add channel dimension (C, H, W) for PyTorch models
        if len(noisy_np.shape) == 2:
            noisy_np = np.expand_dims(noisy_np, axis=0) # Shape: (1, H, W)
            
        noisy_tensor = torch.from_numpy(noisy_np)
        
        # Sanity check finite values
        if not torch.isfinite(noisy_tensor).all():
            raise ValueError(f"NoisyLR tensor contains non-finite values (NaN or Inf) in {filename}")
            
        if self.has_gt:
            # Lazy load Ground Truth image
            gt_path = os.path.join(self.gt_dir, filename)
            gt_np = np.load(gt_path).astype(np.float32)
            
            # Sanity check GT shape
            if gt_np.shape != (256, 256):
                raise ValueError(f"Expected GT shape (256, 256), but got {gt_np.shape} for {filename}")
                
            if len(gt_np.shape) == 2:
                gt_np = np.expand_dims(gt_np, axis=0) # Shape: (1, H, W)
                
            gt_tensor = torch.from_numpy(gt_np)
            
            # Sanity check finite values
            if not torch.isfinite(gt_tensor).all():
                raise ValueError(f"GT tensor contains non-finite values (NaN or Inf) in {filename}")
                
            if self.transform:
                # Apply custom transform if provided
                # Assuming transform can take a tuple of (noisy, gt) or transforms both
                noisy_tensor, gt_tensor = self.transform(noisy_tensor, gt_tensor)
                
            return noisy_tensor, gt_tensor
        else:
            if self.transform:
                noisy_tensor = self.transform(noisy_tensor)
                
            return noisy_tensor

