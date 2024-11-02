from concurrent.futures import ThreadPoolExecutor
import glob
import os
import cv2
from sklearn.model_selection import train_test_split


class ImageResizer:
    def __init__(self, new_size, img_extensions=['jpg', 'png', 'jpeg'], num_workers=2, seed=42):
        """
        Initializes the ImageResizer class with the specified new image size and file extensions.

        Args:
            new_size (tuple): The desired size for the resized images.
            img_extensions (list, optional): A list of supported image file extensions. Defaults to ['jpg', 'png', 'jpeg'].
        """
        self.new_size = new_size
        self.img_extensions = img_extensions
        self.num_workers = num_workers
        self.seed = seed

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

        if not os.path.exists(output_path):
            img = cv2.imread(img_path)
            if img is not None:
                resized_img = cv2.resize(img, self.new_size, interpolation=cv2.INTER_NEAREST_EXACT)
                cv2.imwrite(output_path, resized_img)
                print(f"Image {os.path.basename(img_path)} resized and saved as {output_path}")
            else:
                print(f"Failed to load image {os.path.basename(img_path)}")

    def resize_and_split_parallel(self, images, output_base_folder, subset_name):
        """
        Resizes and saves images into the specified subset folder (train, valid, or test).

        Args:
            images (list): List of image file paths.
            output_base_folder (str): Base folder to save images.
            subset_name (str): Subset folder name (train, valid, or test).
        """
        output_folder = os.path.join(output_base_folder, subset_name)
        os.makedirs(output_folder, exist_ok=True)
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for img_path in images:
                executor.submit(self.resize_and_save, img_path, output_folder)

    def preprocess_images(self, datasets_info, test_size=0.1, valid_size=0.2):
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

            # Define output paths
            output_img_folder = os.path.join(input_img_folder, f'resized_{self.new_size[1]}_{self.new_size[0]}')
            output_mask_folder = os.path.join(input_mask_folder, f'resized_{self.new_size[1]}_{self.new_size[0]}')

            # Find images and masks
            img_files = self.find_files(input_img_folder)
            mask_files = self.find_files(input_mask_folder)

            # Create mapping between image and mask filenames
            img_to_mask = {os.path.basename(img): mask 
                           for img, mask in zip(sorted(img_files), sorted(mask_files))}

            # Split data into train, valid, and test
            train_val_imgs, test_imgs = train_test_split(img_files, test_size=test_size, random_state=self.seed)
            train_imgs, valid_imgs = train_test_split(train_val_imgs, test_size=valid_size / (1 - test_size), random_state=self.seed)

            # Get corresponding mask files
            train_masks = [img_to_mask[os.path.basename(img)] for img in train_imgs]
            valid_masks = [img_to_mask[os.path.basename(img)] for img in valid_imgs]
            test_masks = [img_to_mask[os.path.basename(img)] for img in test_imgs]

            # Resize and save images into respective folders
            self.resize_and_split_parallel(train_imgs, output_img_folder, 'train')
            self.resize_and_split_parallel(valid_imgs, output_img_folder, 'valid')
            self.resize_and_split_parallel(test_imgs, output_img_folder, 'test')

            # Resize and save masks into respective folders
            self.resize_and_split_parallel(train_masks, output_mask_folder, 'train')
            self.resize_and_split_parallel(valid_masks, output_mask_folder, 'valid')
            self.resize_and_split_parallel(test_masks, output_mask_folder, 'test')

            dataset_info['img_path'] = output_img_folder
            dataset_info['mask_path'] = output_mask_folder