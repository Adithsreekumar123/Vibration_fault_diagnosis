# src/ssl/combined_ssl_dataset.py
"""
Combined SSL Dataset for Domain-Invariant Feature Learning

Uses BOTH CWRU and Paderborn data (unlabeled) for self-supervised
contrastive learning. This helps learn features that work across domains.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from src.augment.augmentations import strong_augment


class CombinedSSLDataset(Dataset):
    """
    SSL Dataset that combines source (CWRU) and target (Paderborn) data.
    
    By learning on both domains in a self-supervised way, the encoder
    learns representations that are inherently domain-invariant.
    """
    
    def __init__(
        self,
        source_path="data/processed/cwru_windows.npz",
        target_path="data/processed/paderborn_windows.npz",
        balance=True
    ):
        """
        Args:
            source_path: Path to CWRU data
            target_path: Path to Paderborn data
            balance: If True, undersample larger dataset to balance
        """
        # Load source (CWRU)
        source_data = np.load(source_path)
        self.source_X = source_data["X"]
        
        # Load target (Paderborn)
        target_data = np.load(target_path)
        self.target_X = target_data["X"]
        
        print(f"📊 Source (CWRU) samples: {len(self.source_X)}")
        print(f"📊 Target (Paderborn) samples: {len(self.target_X)}")
        
        # Balance datasets if requested
        if balance:
            min_len = min(len(self.source_X), len(self.target_X))
            
            # Random sample from each
            source_idx = np.random.permutation(len(self.source_X))[:min_len]
            target_idx = np.random.permutation(len(self.target_X))[:min_len]
            
            self.source_X = self.source_X[source_idx]
            self.target_X = self.target_X[target_idx]
            
            print(f"✂️ Balanced to {min_len} samples each")
        
        # Combine both datasets
        self.X = np.concatenate([self.source_X, self.target_X], axis=0)
        
        # Track domain for each sample (0 = source, 1 = target)
        self.domains = np.concatenate([
            np.zeros(len(self.source_X)),
            np.ones(len(self.target_X))
        ])
        
        print(f"✅ Combined SSL dataset: {len(self.X)} total samples")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Returns two augmented views of the same sample for contrastive learning.
        The encoder must learn to recognize them as the same sample.
        """
        x = self.X[idx]
        
        # Create two augmented views
        x1 = strong_augment(x)
        x2 = strong_augment(x)
        
        x1 = torch.tensor(x1, dtype=torch.float32).unsqueeze(0)
        x2 = torch.tensor(x2, dtype=torch.float32).unsqueeze(0)
        
        return x1, x2
