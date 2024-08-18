import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import DroneDataset
from transforms import get_train_transforms, get_val_transforms
from train import train_model
from eval import evaluate_model, plot_predictions
from model import create_model
from utils import load_checkpoint, gc_collect
from early_stopping import EarlyStopping
from optimal_point import plot_furthest_points
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler

def main(args):
    # Constants
    num_classes = 6
    H, W = 384, 576
    batch_size = 8
    num_epochs = 100
    model_path = 'checkpointDeep69Classes6.pt'
    root_dir = 'D:/UniversityFiles/DroneSemantic/SemanticDroneDataset/'
    img_path = os.path.join(root_dir, 'img/')
    mask_path = os.path.join(root_dir, 'mask/')
    num_samples = 1

    # Define color map (cmap) for visualizing class predictions
    cmap = np.array([
        [150, 24, 0],    # Class 0: Остальное (Other)
        [119, 221, 119], # Class 1: Зоны посадки (Landing Zones)
        [58, 117, 196],  # Class 2: Вода (Water)
        [255, 153, 0],   # Class 3: Дороги (Roads)
        [255, 239, 213], # Class 4: Движущиеся объекты (Moving Objects)
        [128, 128, 128]  # Class 5: Маркер (Marker)
    ])

    # Dataset splits
    names = list(map(lambda x: x.replace('.jpg', ''), os.listdir(img_path)))
    X_trainval, X_test = train_test_split(names, test_size=0.1, random_state=19)
    X_train, X_val = train_test_split(X_trainval, test_size=0.2, random_state=19)

    img_train = [os.path.join(img_path, f"{name}.jpg") for name in X_train]
    mask_train = [os.path.join(mask_path, f"{name}.png") for name in X_train]
    img_val = [os.path.join(img_path, f"{name}.jpg") for name in X_val]
    mask_val = [os.path.join(mask_path, f"{name}.png") for name in X_val]
    img_test = [os.path.join(img_path, f"{name}.jpg") for name in X_test]
    mask_test = [os.path.join(mask_path, f"{name}.png") for name in X_test]

    # Class Mapping
    class_mapping = {
        0: 0, 1: 3, 2: 1, 3: 1, 4: 1, 5: 2, 6: 0, 7: 0, 8: 0, 9: 0,
        10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 4, 16: 4, 17: 4, 18: 4,
        19: 0, 20: 0, 21: 5, 22: 0, 23: 0
    }

    # Transforms
    train_transforms = get_train_transforms(H, W)
    val_transforms = get_val_transforms(H, W)

    # Datasets
    train_dataset = DroneDataset(img_train, mask_train, class_mapping, transforms=train_transforms)
    valid_dataset = DroneDataset(img_val, mask_val, class_mapping, transforms=val_transforms)
    test_dataset = DroneDataset(img_test, mask_test, class_mapping, transforms=val_transforms)

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(num_classes).to(device)

    # Optimizer, Scheduler, Scaler, and EarlyStopping
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=10, verbose=True)

    # Load existing model (for evaluation or resuming training)
    if os.path.exists(model_path):
        print("Loading existing model")
        start_epoch, train_losses, val_losses, val_loss_min = load_checkpoint(model, optimizer, scheduler, scaler, model_path)
    else:
        print("No existing model found, starting from scratch")
        start_epoch, train_losses, val_losses, val_loss_min = 0, [], [], [], [], float('inf')

    if args.mode == 'train':
    # Train the model
        train_model(model, train_loader, valid_loader, num_classes, num_epochs, start_epoch, train_losses, val_losses, val_loss_min, device, optimizer, scheduler, scaler, early_stopping, criterion)

    elif args.mode == 'evaluate':
        # Evaluate and plot results
        evaluate_model(model, test_loader, num_classes)
        plot_predictions(test_loader, model, cmap, num_samples)

    if args.mode == 'points':
        # Run the point-finding algorithm
        model.eval()
        with torch.no_grad():
            data = next(iter(test_loader))
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)

            plot_furthest_points(images.cpu(), labels.cpu(), preds.cpu(), cmap, 'marker', 30, 'bottom')

    gc_collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, Evaluate or Run Point-Finding Algorithm")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'points'], required=True, help="Choose whether to train, evaluate, or run the point-finding algorithm")
    
    args = parser.parse_args()
    main(args)
