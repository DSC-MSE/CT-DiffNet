import numpy as np
import torch
from torch.utils.data import Dataset

class SimpleBarrierDataset(Dataset):
    def __init__(self, cache_path: str):
        data = np.load(cache_path)
        voxels = data["voxels"]
        print(f"[Dataset] loaded voxels shape: {voxels.shape}")  # e.g. (N,7,D,D,D)
        self.voxels   = torch.from_numpy(voxels).float()
        self.barriers = torch.from_numpy(data["barriers"].astype(np.float32))
    
    def __len__(self):
        return len(self.barriers)

    def __getitem__(self, idx):
        x = self.voxels[idx]    # (7, D, D, D)
        y = self.barriers[idx]  # scalar
        return x, y
