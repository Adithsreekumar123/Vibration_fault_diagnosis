import torch
from torch.utils.data import Subset
import numpy as np

def create_fewshot_split(dataset, fraction=0.01, seed=42):
    np.random.seed(seed)

    n_total = len(dataset)
    n_train = int(n_total * fraction)

    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return Subset(dataset, train_idx), Subset(dataset, test_idx)
