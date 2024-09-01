import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def get_dataloader(img, patch_size, batch_size):
    ds = PatchDataset(img, patch_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=True), ds

def get_num_patches(img: np.ndarray, patch_size):
    _, w, h = img.shape
    patches_x = w // patch_size
    patches_y = h // patch_size
    total_patches = patches_x * patches_y
    return total_patches, patches_x, patches_y

def split_and_label_img(img: np.ndarray, patch_size):
    # img: (3, w, h)
    total_patches, patches_x, patches_y = get_num_patches(img, patch_size)
    patches = np.empty((total_patches, 3, patch_size, patch_size), dtype=np.float32)
    labels = np.empty(total_patches)
    coords = {} # Map labels to coordinates

    for i in range(patches_x):
        for j in range(patches_y):
            idx = i + j * patches_x
            x_lower, x_upper, y_lower, y_upper = i * patch_size, (i + 1) * patch_size, j * patch_size, (j + 1) * patch_size
            patches[idx, :, :, :] = img[:, x_lower:x_upper, y_lower:y_upper]
            labels[idx] = idx
            coords[idx] = (x_lower, x_upper, y_lower, y_upper)
    
    return patches, labels, coords

class PatchDataset(Dataset):
    def __init__(self, img, patch_size):
        self.img = img
        self.patches, self.labels, self.coords = split_and_label_img(img, patch_size)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patch, label = self.patches[idx], self.labels[idx]
        return patch, label