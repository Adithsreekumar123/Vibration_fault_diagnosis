import numpy as np
import torch
from torch.utils.data import Dataset

class SupervisedDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = data["X"]
        self.y = data["y"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
