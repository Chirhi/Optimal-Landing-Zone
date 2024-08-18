import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(H, W):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=0),
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.5),
        A.GaussNoise(p=0.1),
        A.MotionBlur(p=0.1),
        A.GaussianBlur(p=0.1),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
        A.Resize(height=H, width=W, always_apply=True),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2()
    ])

def get_val_transforms(H, W):
    return A.Compose([
        A.Resize(height=H, width=W, always_apply=True),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2()
    ])
