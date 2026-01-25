import numpy as np
import torch
from torch.utils.data import Dataset
from src.augment.augmentations import strong_augment

class CWRUSSLDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = data["X"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        x1 = strong_augment(x)
        x2 = strong_augment(x)

        x1 = torch.tensor(x1, dtype=torch.float32).unsqueeze(0)
        x2 = torch.tensor(x2, dtype=torch.float32).unsqueeze(0)

        return x1, x2
