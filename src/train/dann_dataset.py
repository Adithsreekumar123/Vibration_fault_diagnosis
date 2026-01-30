# src/train/dann_dataset.py
"""
Dataset for DANN training

Combines source (CWRU) and target (Paderborn) data with domain labels.
- Source (CWRU): Has fault labels + domain label = 0
- Target (Paderborn): No fault labels used + domain label = 1
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class DANNDataset(Dataset):
    """
    Combined dataset for Domain Adversarial Training
    
    Source domain (CWRU): Has class labels, domain = 0
    Target domain (Paderborn): Class labels not used, domain = 1
    """
    
    def __init__(self, source_path, target_path):
        """
        Args:
            source_path: Path to CWRU npz file
            target_path: Path to Paderborn npz file
        """
        # Load source data (CWRU)
        source_data = np.load(source_path)
        self.source_X = source_data["X"]
        self.source_y = source_data["y"]
        
        # Load target data (Paderborn)
        target_data = np.load(target_path)
        self.target_X = target_data["X"]
        self.target_y = target_data["y"]  # Not used in DANN training, but kept for evaluation
        
        self.n_source = len(self.source_X)
        self.n_target = len(self.target_X)
        
        print(f"✅ Loaded DANN dataset:")
        print(f"   Source (CWRU): {self.n_source} samples")
        print(f"   Target (Paderborn): {self.n_target} samples")
    
    def __len__(self):
        # Return max of source and target for balanced sampling
        return max(self.n_source, self.n_target)
    
    def __getitem__(self, idx):
        """
        Returns a pair of source and target samples
        
        Returns:
            source_x: Source domain signal (1, 4096)
            source_y: Source class label (0-3)
            source_domain: 0 (source)
            target_x: Target domain signal (1, 4096)
            target_domain: 1 (target)
        """
        # Get source sample (with wrapping if needed)
        source_idx = idx % self.n_source
        source_x = torch.tensor(self.source_X[source_idx], dtype=torch.float32).unsqueeze(0)
        source_y = torch.tensor(self.source_y[source_idx], dtype=torch.long)
        source_domain = torch.tensor(0, dtype=torch.float32)
        
        # Get target sample (with wrapping if needed)
        target_idx = idx % self.n_target
        target_x = torch.tensor(self.target_X[target_idx], dtype=torch.float32).unsqueeze(0)
        target_domain = torch.tensor(1, dtype=torch.float32)
        
        return source_x, source_y, source_domain, target_x, target_domain
