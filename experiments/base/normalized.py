import numpy as np
import torch
from torch.utils.data.dataset import ConcatDataset, Subset, TensorDataset
from src.algos import util

class NormalizedTensorDataset(TensorDataset):
    def __init__(self, data, targets, data_mean=None, data_std=None, target_mean=None, target_std=None):
        self.data_mean = data.mean(dim=0) if data_mean is None else data_mean
        self.data_std = data.std(dim=0) if data_std is None else data_std
        normalized_data = torch.nan_to_num((data - self.data_mean) / self.data_std)

        self.target_mean = targets.mean(dim=0) if target_mean is None else target_mean
        self.target_std = targets.std(dim=0) if target_std is None else target_std
        normalized_targets = torch.nan_to_num((targets - self.target_mean) / self.target_std)

        super().__init__(normalized_data, normalized_targets)