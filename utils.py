import os
import glob
import numpy as np
import torch
import gc
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import DroneDataset
from transforms import get_train_transforms, get_val_transforms
from torch.utils.data import ConcatDataset
import cv2
from concurrent.futures import ThreadPoolExecutor

class ImageResizer:
    def __init__(self, new_size, img_extensions=['jpg', 'png', 'jpeg'], num_workers=2):
        """
        Initializes the ImageResizer class with the specified new image size and file extensions.

        Args:
            new_size (tuple): The desired size for the resized images.
            img_extensions (list, optional): A list of supported image file extensions. Defaults to ['jpg', 'png', 'jpeg'].
        """
        self.new_size = new_size
        self.img_extensions = img_extensions
        self.num_workers = num_workers

    def find_files(self, dir_path):
        """
        Finds files in the specified directory with the given extensions.

        Args:
            dir_path (str): Directory path.

        Returns:
            list: List of file paths.
        """
        files = []
        for ext in self.img_extensions:
            files.extend(glob.glob(os.path.join(dir_path, f'*.{ext}')))
        return files

    def resize_and_save(self, img_path, output_folder):
        """
        Resizes an image and saves it to the specified output folder.

        Args:
            img_path (str): The path to the image file to be resized.
            output_folder (str): The folder where the resized image will be saved.

        Notes:
            If the output path does not exist, it will be created.
        """
        output_path = os.path.join(output_folder, os.path.basename(img_path))

        # Check if the output path exists and create it if it doesn't
        if not os.path.exists(output_path):
            img = cv2.imread(img_path)
            if img is not None:
                resized_img = cv2.resize(img, self.new_size, interpolation=cv2.INTER_NEAREST_EXACT)

                cv2.imwrite(output_path, resized_img)
                print(f"Image {os.path.basename(img_path)} resized and saved as {output_path}")
            else:
                print(f"Failed to load image {os.path.basename(img_path)}")

    def resize_images(self, input_folder, output_folder):
        """
        Resizes images in a specified input folder and saves them to a specified output folder.

        Args:
            input_folder (str): The path to the folder containing the images to be resized.
            output_folder (str): The path to the folder where the resized images will be saved.
        """
        os.makedirs(output_folder, exist_ok=True)
        image_files = self.find_files(input_folder)

        # Use ThreadPoolExecutor to resize images in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for img_path in image_files:
                executor.submit(self.resize_and_save, img_path, output_folder)

    def preprocess_images(self, datasets_info):
        """
        Preprocesses images in the specified datasets by resizing them and updating their paths.

        Args:
            datasets_info (list): A list of dictionaries containing information about the datasets.
                Each dictionary should have the following keys:
                    - 'img_path': The path to the folder containing the images to be resized.
                    - 'mask_path': The path to the folder containing the masks to be resized.
        """
        for dataset_info in datasets_info:
            input_img_folder = dataset_info['img_path']
            input_mask_folder = dataset_info['mask_path']

            output_img_folder = os.path.join(input_img_folder, f'resized_{self.new_size[1]}_{self.new_size[0]}')
            output_mask_folder = os.path.join(input_mask_folder, f'resized_{self.new_size[1]}_{self.new_size[0]}')

            self.resize_images(input_img_folder, output_img_folder)
            self.resize_images(input_mask_folder, output_mask_folder)

            dataset_info['img_path'] = output_img_folder
            dataset_info['mask_path'] = output_mask_folder

def remap_labels(mask, class_mapping):
    """
    Remaps the labels in a given mask using a provided class mapping.

    Args:
        mask (numpy.ndarray): The input mask to be remapped.
        class_mapping (dict): A dictionary mapping old class labels to new class labels.

    Returns:
        numpy.ndarray: The remapped mask with the labels replaced according to the class mapping.
    """
    new_mask = np.zeros_like(mask)
    for old_class, new_class in class_mapping.items():
        new_mask[mask == old_class] = new_class
    return new_mask

def load_checkpoint(model, optimizer, scheduler, scaler, filename='checkpoint.pt'):
    """
    Load a checkpoint from a file and restore the model, optimizer, scheduler, and scaler.

    Args:
        model (torch.nn.Module): The model to be restored.
        optimizer (torch.optim.Optimizer): The optimizer to be restored.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to be restored.
        scaler (torch.cuda.amp.GradScaler): The gradient scaler to be restored.
        filename (str, optional): The name of the file to load the checkpoint from. Defaults to 'checkpoint.pt'.

    Returns:
        tuple: A tuple containing the epoch, training losses, validation losses, and the minimum validation loss.
    """
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

# Function to denormalize images
def denormalize(image, mean, std):
    """
    Denormalizes an image by reversing the normalization process in transforms module.

    Args:
        image (torch.Tensor): The normalized image tensor.
        mean (list or tuple): The mean values used for normalization.
        std (list or tuple): The standard deviation values used for normalization.

    Returns:
        np.ndarray: The denormalized image.
    """
    mean = np.array(mean)
    std = np.array(std)
    image = image.permute(1, 2, 0).numpy()
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image

def plot_loss(train_losses, val_losses):
    """
    Plots the training and validation loss over epochs.

    Args:
        train_losses (list): A list of training losses at each epoch.
        val_losses (list): A list of validation losses at each epoch.
    """
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
    def __init__(self, datasets_info, batch_size, num_workers=0, pin_memory=True, persistent_workers=True, prefetch_factor=2):
        """
        Initializes a MultiDatasetLoader instance.

        Args:
            datasets_info (list): A list of dictionaries containing information about each dataset.
            H (int): The height for resizing images.
            W (int): The width for resizing images.
            batch_size (int): The batch size for the DataLoader.
            num_workers (int): Number of subprocesses to use for data loading.
            pin_memory (bool): Whether to use pinned (page-locked) memory.
        """
        self.datasets_info = datasets_info
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

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

    def create_dataloader(self, dataset, shuffle):
        """
        Creates a DataLoader for the given dataset.

        Args:
            dataset (Dataset): The dataset to create a DataLoader for.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            DataLoader: The DataLoader for the given dataset.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor
        )

    def create_loaders(self):
        """
        Creates data loaders for training, validation, and testing datasets.
        Iterates over the provided dataset information, finds common image and mask files,
        splits them into training, validation, and testing sets, applies transformations,
        and creates data loaders for each set.
        Returns:
            dict: A dictionary containing data loaders for each dataset, with keys in the format
                  '{dataset_name}_{set}_loader' (e.g., 'dataset1_train_loader').
        """
        loaders = {}
        for dataset_info in self.datasets_info:
            name = dataset_info['name']
            img_path = dataset_info['img_path']
            mask_path = dataset_info['mask_path']
            class_mapping = dataset_info['class_mapping']
            mask_type = dataset_info.get('mask_type', 'grayscale')

            # Find images and masks with different extensions
            img_extensions = ['jpg', 'png', 'jpeg']

            img_files = self.find_files(img_path, img_extensions)
            mask_files = self.find_files(mask_path, img_extensions)

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
            train_transforms = get_train_transforms()
            val_transforms = get_val_transforms()

            # Datasets
            train_dataset = DroneDataset(img_train, mask_train, class_mapping, transforms=train_transforms, mask_type=mask_type)
            valid_dataset = DroneDataset(img_val, mask_val, class_mapping, transforms=val_transforms, mask_type=mask_type)
            test_dataset = DroneDataset(img_test, mask_test, class_mapping, transforms=val_transforms, mask_type=mask_type)

            # Data Loaders
            loaders[f'{name}_train_loader'] = self.create_dataloader(train_dataset, shuffle=True)
            loaders[f'{name}_valid_loader'] = self.create_dataloader(valid_dataset, shuffle=False)
            loaders[f'{name}_test_loader'] = self.create_dataloader(test_dataset, shuffle=False)

        return loaders

    def create_combined_loaders(self):
        """
        Combines data loaders from different datasets into a single loader.
        Concatenates the training, validation, and testing datasets from each loader,
        and creates new data loaders for the combined datasets.

        Returns:
            dict: A dictionary containing the combined data loaders, with keys
                  'train_loader', 'valid_loader', and 'test_loader'.
        """
        loaders = self.create_loaders()

        combined_train_dataset = ConcatDataset([loaders[key].dataset for key in loaders if 'train_loader' in key])
        combined_valid_dataset = ConcatDataset([loaders[key].dataset for key in loaders if 'valid_loader' in key])
        combined_test_dataset = ConcatDataset([loaders[key].dataset for key in loaders if 'test_loader' in key])

        combined_train_loader = self.create_dataloader(combined_train_dataset, shuffle=True)
        combined_valid_loader = self.create_dataloader(combined_valid_dataset, shuffle=False)
        combined_test_loader = self.create_dataloader(combined_test_dataset, shuffle=False)

        return {
            'train_loader': combined_train_loader,
            'valid_loader': combined_valid_loader,
            'test_loader': combined_test_loader
        }

def measure_inference_time(model, device, input_size=(1, 3, 224, 224), num_iterations=100):
    """
    Measures the average inference time of a given model on a specified device.

    Args:
        model: The model to measure inference time for.
        device: The device to run the model on.
        input_size (tuple, optional): The size of the input tensor. Defaults to (1, 3, 224, 224).
        num_iterations (int, optional): The number of iterations to run the model for. Defaults to 100.
        find_points_func (function, optional): The function for finding points. Should accept images and predictions.

    Returns:
        float: The average inference time of the model.
    """

    # Create dummy input
    dummy_input = torch.randn(input_size).to(device)
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)
    
    # Measure inference time
    start_time = time.perf_counter()  # Use perf_counter for better accuracy
    
    with torch.no_grad():
        for _ in range(num_iterations):
            model(dummy_input)
    
    end_time = time.perf_counter()
    
    avg_inference_time = (end_time - start_time) / num_iterations
    
    return avg_inference_time
