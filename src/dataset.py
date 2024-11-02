import cv2
import torch
from torch.utils.data import Dataset

class DroneDataset(Dataset):
    def __init__(self, images, masks, class_mapping, transforms=None, placeholder_class=None):
        """
        Initializes a DroneDataset instance.

        Args:
            images (list): A list of image file paths.
            masks (list): A list of mask file paths corresponding to the images.
            class_mapping (dict): A dictionary mapping old class labels to new class labels.
            transforms (callable, optional): An optional transform function to apply to the images and masks. Defaults to None.
            placeholder_class (int, optional): The class label to use for placeholder data. Defaults to None.
        """
        self.images = images
        self.masks = masks
        self.class_mapping = class_mapping
        self.transforms = transforms
        self.placeholder_class = placeholder_class

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding mask from the dataset at the specified index.
        """
        from utils import remap_labels
        image = cv2.imread(self.images[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = remap_labels(mask, self.class_mapping, placeholder_class=self.placeholder_class)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
