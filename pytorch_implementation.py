import torch
from core import *
from torch.utils.data import Dataset


class SpatialPointVectorDataset(Dataset):
    def __init__(self, sdv):
        self.SpatialPointVector = sdv

    def __len__(self):
        return len(SpatialPointVector)

    def __getitem__(self, item):
        return torch.from_numpy(SpatialPointVector[item])

