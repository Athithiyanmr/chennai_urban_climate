import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class BuiltupDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.imgs = sorted(Path(img_dir).glob("*.npy"))
        self.masks = sorted(Path(mask_dir).glob("*.npy"))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x = np.load(self.imgs[idx], allow_pickle=False).astype("float32")
        y = np.load(self.masks[idx], allow_pickle=False).astype("float32")

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        )