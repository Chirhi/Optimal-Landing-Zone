import os
import glob
import numpy as np
import torch
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import DroneDataset
from transforms import get_train_transforms, get_val_transforms

def remap_labels(mask, class_mapping):
    new_mask = np.zeros_like(mask)
    for old_class, new_class in class_mapping.items():
        new_mask[mask == old_class] = new_class
    return new_mask

def load_checkpoint(model, optimizer, scheduler, scaler, filename='checkpoint.pt'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    if val_losses:
        val_loss_min = min(val_losses)
    else:
        print("Warning: val_losses is empty, setting val_loss_min to infinity")
        val_loss_min = float('inf')

    print(f"Loaded checkpoint: Epoch: {epoch}, Validation Loss: {val_loss_min:.6f}")
    return epoch, train_losses, val_losses, val_loss_min

def plot_loss(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()

class MultiDatasetLoader:
    def __init__(self, datasets_info, H, W, batch_size):
        """
        Initializes a MultiDatasetLoader instance.

        Args:
            datasets_info (list): A list of dictionaries containing information about each dataset.
            H (int): The height for resizing images.
            W (int): The width for resizing images.
            batch_size (int): The batch size for the DataLoader.

        Returns:
            None
        """
        self.datasets_info = datasets_info
        self.H = H
        self.W = W
        self.batch_size = batch_size

    def find_files(self, dir_path, extensions):
        """
        Finds files in the specified directory with the given extensions.

        Args:
            dir_path (str): Directory path.
            extensions (list): List of file extensions to search for.

        Returns:
            list: List of file paths.
        """
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(dir_path, f'*.{ext}')))
        return files

    def create_loaders(self):
        loaders = {}
        for dataset_info in self.datasets_info:
            name = dataset_info['name']
            img_path = dataset_info['img_path']
            mask_path = dataset_info['mask_path']
            class_mapping = dataset_info['class_mapping']
            mask_type = dataset_info.get('mask_type', 'grayscale')

            # Find images and masks with different extensions
            img_extensions = ['jpg', 'png', 'jpeg']
            mask_extensions = ['png', 'jpg']

            img_files = self.find_files(img_path, img_extensions)
            mask_files = self.find_files(mask_path, mask_extensions)

            # Extract file names without extensions for splitting
            img_names = [os.path.splitext(os.path.basename(f))[0] for f in img_files]
            mask_names = [os.path.splitext(os.path.basename(f))[0] for f in mask_files]

            # Find common files between images and masks
            common_names = list(set(img_names).intersection(mask_names))

            # Dataset splits
            X_trainval, X_test = train_test_split(common_names, test_size=0.1, random_state=19)
            X_train, X_val = train_test_split(X_trainval, test_size=0.2, random_state=19)

            img_train = [next(f for f in img_files if os.path.basename(f).startswith(name)) for name in X_train]
            mask_train = [next(f for f in mask_files if os.path.basename(f).startswith(name)) for name in X_train]
            img_val = [next(f for f in img_files if os.path.basename(f).startswith(name)) for name in X_val]
            mask_val = [next(f for f in mask_files if os.path.basename(f).startswith(name)) for name in X_val]
            img_test = [next(f for f in img_files if os.path.basename(f).startswith(name)) for name in X_test]
            mask_test = [next(f for f in mask_files if os.path.basename(f).startswith(name)) for name in X_test]

            # Transforms
            train_transforms = get_train_transforms(self.H, self.W)
            val_transforms = get_val_transforms(self.H, self.W)

            # Datasets
            train_dataset = DroneDataset(img_train, mask_train, class_mapping, transforms=train_transforms, mask_type=mask_type)
            valid_dataset = DroneDataset(img_val, mask_val, class_mapping, transforms=val_transforms, mask_type=mask_type)
            test_dataset = DroneDataset(img_test, mask_test, class_mapping, transforms=val_transforms, mask_type=mask_type)

            # Data Loaders
            loaders[f'{name}_train_loader'] = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
            loaders[f'{name}_valid_loader'] = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
            loaders[f'{name}_test_loader'] = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        return loaders

def create_dataloader(dataset, batch_size, shuffle, num_workers=0, pin_memory=True):
    """
    Creates a DataLoader for the given dataset.

    Args:
        dataset (Dataset): The dataset to create a DataLoader for.
        batch_size (int): The batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int, optional): Number of subprocesses to use for data loading. Default is 0.
        pin_memory (bool, optional): Whether to use pinned (page-locked) memory. Default is True.

    Returns:
        DataLoader: The DataLoader for the given dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )