import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_dimensions):
    """
    A.RandomResizedCrop        # Simulates footage from different drone flight altitudes
    A.HorizontalFlip           # Trains model to recognize landing spots regardless of approach direction
    A.VerticalFlip             # Additional invariance to image orientation
    A.Rotate                   # Simulates different drone approach angles to landing site
    A.ShiftScaleRotate         # Combined perspective changes during maneuvering
    A.RandomBrightnessContrast # Adaptation to different times of day and lighting
    A.RandomGamma              # Correction for different camera exposure conditions
    A.CLAHE                    # Improves detail visibility in dark/bright areas
    A.ElasticTransform         # Simulates distortions from air mass movement
    A.GridDistortion           # Simulates drone camera optical distortions
    A.OpticalDistortion        # Simulates camera lens distortions
    A.RandomShadow             # Trains model to work with shadows from clouds and objects
    A.ColorJitter              # Adaptation to different color shooting conditions
    A.Perspective              # Simulates different viewing angles during approach
    A.HueSaturationValue       # Adaptation to different lighting and weather conditions
    A.GaussNoise               # Simulates camera sensor noise
    A.MotionBlur               # Simulates blur from drone movement
    A.GaussianBlur             # Simulates camera defocus
    A.CoarseDropout            # Simulates partial view obstruction (drops, dust)
    A.Normalize                # Standardizes data for training based on ImageNet statistics
    ToTensorV2                 # Converts to PyTorch tensors
    """
    height, width = image_dimensions

    return A.Compose([
        A.RandomResizedCrop(size=(height, width), scale=(0.5, 1), ratio=(0.75, 1.33), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate((-90, 90), p=0.5),
        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30, shift_limit=0.1, p=0.5, border_mode=0),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3), 
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.2),
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2),
        A.GaussNoise(p=0.1),
        A.MotionBlur(p=0.1),
        A.GaussianBlur(p=0.1),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
