import os
import argparse
import torch
import numpy as np
from train import train_model
from eval import evaluate_model, plot_predictions
from model import create_model
from utils import load_checkpoint, gc_collect, MultiDatasetLoader, plot_loss, create_dataloader
from early_stopping import EarlyStopping
from optimal_point import plot_furthest_points
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler
from torch.utils.data import ConcatDataset

def main(args):
    """
    This is the main function that controls the entire program flow. It takes in a single argument 'args' which is expected to be a namespace object containing the mode of operation.

    The function supports three modes of operation:
    - 'train': This mode trains a deep learning model on a combined dataset of images and their corresponding masks.
    - 'evaluate': This mode evaluates the performance of a trained model on a test dataset and plots the results.
    - 'points': This mode runs a point-finding algorithm on a test dataset using a trained model.

    The function returns no value.

    Parameters:
    args (namespace): A namespace object containing the mode of operation.

    Returns:
    None
    """
    # Constants
    num_classes = 6              # number of classes
    H, W = 384, 576              # image height and width to resize the images
    batch_size = 8               # batch size
    num_epochs = 100             # number of epochs to train the model
    model_path = 'checkpoint.pt' # file path to the model checkpoint
    num_samples = 1              # number of samples to plot

    # Define color map (cmap) for visualizing class predictions
    cmap = np.array([
        [150, 24, 0],    # Class 0: Остальное (Other)
        [119, 221, 119], # Class 1: Зоны посадки (Landing Zones)
        [58, 117, 196],  # Class 2: Вода (Water)
        [255, 153, 0],   # Class 3: Дороги (Roads)
        [255, 239, 213], # Class 4: Движущиеся объекты (Moving Objects)
        [128, 128, 128]  # Class 5: Маркер (Marker)
    ])

    datasets_info = [
        {
            'name': 'semantic_drone_dataset',
            'img_path': 'D:/UniversityFiles/DroneSemantic/SemanticDroneDataset/img/',
            'mask_path': 'D:/UniversityFiles/DroneSemantic/SemanticDroneDataset/mask/',
            'class_mapping': {
            0: 0, 1: 3, 2: 1, 3: 1, 4: 1, 5: 2, 6: 0, 7: 0, 8: 0, 9: 0,
            10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 4, 16: 4, 17: 4, 18: 4,
            19: 0, 20: 0, 21: 5, 22: 0, 23: 0
            },
            'mask_type': 'grayscale'
        },
        {
            'name': 'swiss_okuna',
            'img_path': 'D:/UniversityFiles/DroneSemantic/SwissOkuna/img',
            'mask_path': 'D:/UniversityFiles/DroneSemantic/SwissOkuna/mask',
            'class_mapping': {0: 1, 1: 0, 2: 0, 3: 3, 4: 1, 5: 0, 6: 0, 7: 4, 8: 2, 9: 4},
            'mask_type': 'grayscale'
        }
    ]

    multi_loader = MultiDatasetLoader(datasets_info, H, W, batch_size)
    loaders = multi_loader.create_loaders()

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(num_classes).to(device)

    # Optimizer, Scheduler, Scaler, and EarlyStopping
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)
    scaler = GradScaler('cuda')

    # Load existing model (for evaluation or resuming training)
    if os.path.exists(model_path):
        print("Loading existing model")
        start_epoch, train_losses, val_losses, val_loss_min = load_checkpoint(model, optimizer, scheduler, scaler, model_path)
    else:
        print("No existing model found, starting from scratch")
        start_epoch, train_losses, val_losses, val_loss_min = 0, [], [], float('inf')

    early_stopping = EarlyStopping(patience=10, verbose=True, val_loss_min=val_loss_min)

    # Collect all the data in a single dataset
    combined_train_dataset = ConcatDataset([loaders[key].dataset for key in loaders if 'train_loader' in key])
    combined_valid_dataset = ConcatDataset([loaders[key].dataset for key in loaders if 'valid_loader' in key])
    combined_test_dataset = ConcatDataset([loaders[key].dataset for key in loaders if 'test_loader' in key])

    # Create a data loader for the combined dataset
    combined_train_loader = create_dataloader(combined_train_dataset, batch_size, shuffle=True)
    combined_valid_loader = create_dataloader(combined_valid_dataset, batch_size, shuffle=False)
    combined_test_loader = create_dataloader(combined_test_dataset, batch_size, shuffle=False)

    # Train the model on the combined dataset
    if args.mode == 'train':
        train_model(
            model, 
            combined_train_loader, 
            combined_valid_loader, 
            num_classes, 
            num_epochs, 
            start_epoch, 
            train_losses, 
            val_losses, 
            val_loss_min, 
            device, 
            optimizer, 
            scheduler, 
            scaler, 
            early_stopping, 
            criterion
        )

    # Evaluate and plot results
    elif args.mode == 'evaluate':
        evaluate_model(model, combined_test_loader, num_classes)
        plot_predictions(combined_test_loader, model, cmap, num_samples)

        if train_losses and val_losses:
            plot_loss(train_losses, val_losses)
        else:
            print("No loss data available to plot.")

    if args.mode == 'points':
        # Run the point-finding algorithm
        model.eval()
        with torch.no_grad():
            data = next(iter(combined_test_loader))
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)

            plot_furthest_points(images.cpu(), labels.cpu(), preds.cpu(), cmap, 'marker', 30, 'bottom', num_samples)

    gc_collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, Evaluate or Run Point-Finding Algorithm")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'points'], required=True, help="Choose whether to train, evaluate, or run the point-finding algorithm")
    
    args = parser.parse_args()
    main(args)
