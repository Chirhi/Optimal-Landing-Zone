import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class DroneDataset(Dataset):
    def __init__(self, images, masks, class_mapping, transforms=None, mask_type='grayscale'):
        """
        Initializes a DroneDataset instance.

        Args:
            images (list): A list of image file paths.
            masks (list): A list of mask file paths corresponding to the images.
            class_mapping (dict): A dictionary mapping old class labels to new class labels.
            transforms (callable, optional): An optional transform function to apply to the images and masks. Defaults to None.
            mask_type (str, optional): The type of masks ('grayscale' or 'color'). Defaults to 'grayscale'.

        Returns:
            None
        """
        self.images = images
        self.masks = masks
        self.class_mapping = class_mapping
        self.transforms = transforms
        self.mask_type = mask_type

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        from utils import remap_labels
        image = cv2.imread(self.images[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mask_type == 'grayscale':
            mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        else:
            mask = cv2.imread(self.masks[idx], cv2.IMREAD_COLOR)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if self.mask_type == 'grayscale':
            mask = remap_labels(mask, self.class_mapping)
            mask = torch.tensor(mask, dtype=torch.long)
        else:
            # Применяем remap_labels только к каналам маски
            mask = remap_labels(mask[:, :, 0], self.class_mapping)
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
