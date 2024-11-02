from transforms import get_train_transforms, get_val_transforms
from dataset import DroneDataset
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
import glob
import os

class MultiDatasetLoader:
    def __init__(self, datasets_info, batch_size, num_workers=0, image_dimensions=(224, 224), pin_memory=True, persistent_workers=True, prefetch_factor=2):
        """
        Initializes a MultiDatasetLoader instance.

        Args:
            datasets_info (list): A list of dictionaries containing information about each dataset.
            image_dimensions (tuple): A tuple containing the height and width for resizing images.
            batch_size (int): The batch size for the DataLoader.
            num_workers (int): Number of subprocesses to use for data loading.
            pin_memory (bool): Whether to use pinned (page-locked) memory.
            persistent_workers (bool): Whether to use persistent workers.
            prefetch_factor (int): The prefetch factor for the DataLoader.
        """
        self.datasets_info = datasets_info
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.image_dimensions = image_dimensions

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
            img_base_path = dataset_info['img_path']
            mask_base_path = dataset_info['mask_path']
            class_mapping = dataset_info['class_mapping']
            placeholder_class = dataset_info.get('placeholder_class', None)

            # Define paths to train, valid, and test folders for images and masks
            paths = {
                'train': (os.path.join(img_base_path, 'train'), os.path.join(mask_base_path, 'train')),
                'valid': (os.path.join(img_base_path, 'valid'), os.path.join(mask_base_path, 'valid')),
                'test': (os.path.join(img_base_path, 'test'), os.path.join(mask_base_path, 'test'))
            }

            # Create datasets for train, valid, and test
            datasets = {
                split: DroneDataset(
                    images=[os.path.join(paths[split][0], fname) for fname in os.listdir(paths[split][0]) if fname.endswith(('jpg', 'png', 'jpeg'))],
                    masks=[os.path.join(paths[split][1], fname) for fname in os.listdir(paths[split][1]) if fname.endswith(('jpg', 'png', 'jpeg'))],
                    class_mapping=class_mapping,
                    transforms=get_train_transforms(self.image_dimensions) if split == 'train' else get_val_transforms(),
                    placeholder_class=placeholder_class
                )
                for split in ['train', 'valid', 'test']
            }

            # Create DataLoaders for each split
            loaders[f'{name}_train_loader'] = self.create_dataloader(datasets['train'], shuffle=True)
            loaders[f'{name}_valid_loader'] = self.create_dataloader(datasets['valid'], shuffle=False)
            loaders[f'{name}_test_loader'] = self.create_dataloader(datasets['test'], shuffle=False)

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

        # Check if loaders are empty
        if not loaders:
            print("No valid datasets found for loading.")
            return None

        combined_train_dataset = ConcatDataset([loaders[key].dataset for key in loaders if 'train_loader' in key])
        combined_valid_dataset = ConcatDataset([loaders[key].dataset for key in loaders if 'valid_loader' in key])
        combined_test_dataset = ConcatDataset([loaders[key].dataset for key in loaders if 'test_loader' in key])

        # Print combined dataset sizes
        print("\nCombined Dataset Sizes")
        print(f"Total training images:   {len(combined_train_dataset)}")
        print(f"Total validation images: {len(combined_valid_dataset)}")
        print(f"Total test images:       {len(combined_test_dataset)}")

        # Check if combined datasets are empty
        if len(combined_train_dataset) == 0 or len(combined_valid_dataset) == 0 or len(combined_test_dataset) == 0:
            print("One or more combined datasets are empty after filtering. Check dataset paths and contents.")
            return None

        combined_train_loader = self.create_dataloader(combined_train_dataset, shuffle=True)
        combined_valid_loader = self.create_dataloader(combined_valid_dataset, shuffle=False)
        combined_test_loader = self.create_dataloader(combined_test_dataset, shuffle=False)

        return {
            'train_loader': combined_train_loader,
            'valid_loader': combined_valid_loader,
            'test_loader': combined_test_loader
        }