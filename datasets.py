import torch
from torch.utils.data import Dataset

import os
import numpy as np


class LatentDataset(Dataset):
    def __init__(self, latent_dir):
        self.latent_dir = latent_dir
        self.latent_files = sorted(os.listdir(latent_dir))

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_file = self.latent_files[idx]
        latent = np.load(os.path.join(self.latent_dir, latent_file))
        return torch.tensor(latent)
