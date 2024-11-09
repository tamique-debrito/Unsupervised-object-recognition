from enum import Enum
from matplotlib import patches
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class PredictionType(Enum):
    PATCH_INDEX = 1
    COORD = 2
    CLUSTERED_COORD = 3
    CLOSE_TO_TARGET = 4

def is_coord_pred_type(prediction_type: PredictionType):
    if prediction_type == PredictionType.COORD or prediction_type == PredictionType.CLUSTERED_COORD: return True
    return False

def is_cat_pred_type(prediction_type: PredictionType):
    if prediction_type == PredictionType.PATCH_INDEX or prediction_type == PredictionType.CLOSE_TO_TARGET: return True
    return False

def get_dataloader(img, batch_size, prediction_type: PredictionType, patch_size, center = None, radius = None, stride = None):
    if prediction_type == PredictionType.CLOSE_TO_TARGET:
        dataset = PatchDatasetByProx(img, patch_size, stride, center, radius)
    else:
        dataset = PatchDataset(img, patch_size, prediction_type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), dataset

def get_num_patches(img: np.ndarray, patch_size, stride):
    _, x, y = img.shape
    patches_x = (x - patch_size) // stride + 1
    patches_y = (y - patch_size) // stride + 1
    total_patches = patches_x * patches_y
    return total_patches, patches_x, patches_y

def split_and_label_img(img: np.ndarray, patch_size, stride=None):
    # img: (3, w, h)
    if stride is None:
        stride = patch_size
    total_patches, patches_x, patches_y = get_num_patches(img, patch_size, stride)
    _, x, y = img.shape
    patches = np.empty((total_patches, 3, patch_size, patch_size), dtype=np.float32)
    labels = np.empty(total_patches)
    # Map index to patch boundary and coordinates
    patch_boundaries = {}
    coords = {}

    for i in range(patches_x):
        for j in range(patches_y):
            idx = i + j * patches_x
            x_lower, x_upper, y_lower, y_upper = i * stride, i * stride + patch_size, j * stride, j * stride + patch_size
            patches[idx, :, :, :] = img[:, x_lower:x_upper, y_lower:y_upper]
            labels[idx] = idx
            patch_boundaries[idx] = (x_lower, x_lower + stride, y_lower, y_lower + stride)
            coords[idx] = np.array(((x_lower + x_upper) / 2, (y_lower + y_upper) / 2), dtype=np.float32)

    patches = patches / 255.0

    return patches, labels, patch_boundaries, coords

class PatchDataset(Dataset):
    def __init__(self, img, patch_size, stride = None, prediction_type: PredictionType = PredictionType.PATCH_INDEX, use_aug = False):
        self.img = img
        self.patches, self.patch_index, self.patch_center_bounds, self.coords = split_and_label_img(img, patch_size)
        self.prediction_type = prediction_type
        self.use_aug = use_aug
        self.cluster_map = {idx: idx for idx in self.patch_index}
        self.use_cluster = False

    def __len__(self):
        return len(self.patch_index)
    
    def set_cluster_for_sample(self, location, cluster):
        self.cluster_map[location] = cluster

    def __getitem__(self, idx):
        if self.prediction_type == PredictionType.PATCH_INDEX:
            patch, location = self.patches[idx], self.patch_index[idx]
            if self.use_cluster:
                location = self.cluster_map[location]
        elif is_coord_pred_type(self.prediction_type): patch, location = self.patches[idx], self.coords[idx]

        if self.use_aug: patch = patch + np.array(np.random.uniform(0.0, 0.05, patch.shape), dtype=np.float32)

        return patch, location

def get_label(center, radius, coord):
    return 1 if (np.power(center - coord, 2).sum() <= radius ** 2) else 0

class PatchDatasetByProx(Dataset):
    def __init__(self, img, patch_size, stride, center, radius):
        self.center = np.array(center)
        self.patch_size = patch_size
        self.radius = radius
        self.stride = stride
        self.img = img
        self.patches, self.patch_index, self.patch_center_bounds, self.coords = split_and_label_img(img, patch_size, stride)
        self.labels = np.zeros(self.patches.shape[0])
        for idx in self.coords:
            coord = self.coords[idx]
            self.labels[idx] = get_label(self.center, self.radius, coord)

    def update_image(self, new_img):
        # Swap out the stored image out for the new one, and re-extract patches based off the new image. Patch generation parameters and patch labels stay the same
        self.img = new_img
        self.patches, _, _, _ = split_and_label_img(self.img, self.patch_size, self.stride)

    def __len__(self):
        return len(self.patch_index)

    def __getitem__(self, idx):
        patch, label = self.patches[idx], self.labels[idx]
        #patch = patch + np.array(np.random.uniform(0.0, 0.1, patch.shape), dtype=np.float32)

        return patch, label