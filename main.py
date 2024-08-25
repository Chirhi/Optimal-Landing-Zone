import os
import argparse
import torch
import numpy as np
from train import train_model
from eval import evaluate_model, plot_predictions
from model import create_model
from utils import load_checkpoint, gc_collect, MultiDatasetLoader, plot_loss, ImageResizer, measure_inference_time
from early_stopping import EarlyStopping
from optimal_point import plot_furthest_points
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler

def main(args):
    """
    This is the main function that controls the entire program flow. It takes in a single argument 'args' which is expected to be a namespace object containing the mode of operation.

    The function supports three modes of operation:
    - 'train': This mode trains a deep learning model on a combined dataset of images and their corresponding masks.
    - 'evaluate': This mode evaluates the performance of a trained model on a test dataset and plots the results.
    - 'points': This mode runs a point-finding algorithm on a test dataset using a trained model.
    """
    # Constants
    num_classes = 6                     # number of classes
    H, W = 384, 576                     # image height and width to resize the images
    batch_size = 25                     # batch size
    num_epochs = 200                    # number of epochs to train the model
    model_path = 'final_model.pt'       # file path to the model checkpoint
    num_samples = 1                     # number of samples to plot for evaluation
    num_workers = 4                     # number of workers to use for data loading
    zone_type = 'marker'                # type of zone to generate the mask for
    view_mode = 'bottom'                # view mode of the camera
    num_points = 30                     # max number of points to find in the specified zone

    # Define color map (cmap) for visualizing class predictions
    cmap = np.array([
        [150, 24, 0],    # Class 0: Остальное (Other)
        [119, 221, 119], # Class 1: Зоны посадки (Landing Zones)
        [58, 117, 196],  # Class 2: Вода (Water)
        [255, 153, 0],   # Class 3: Дороги (Roads)
        [255, 239, 213], # Class 4: Движущиеся объекты (Moving Objects)
        [128, 128, 128]  # Class 5: Маркер (Marker)
    ])

    # Define datasets information
    datasets_info = [
        {
            'name': 'semantic_drone_dataset',
            'img_path': 'C:/Users/zephyr/JupyterNotebook/NIRDiploma/DroneSemantic/SemanticDroneDataset/img/',
            'mask_path': 'C:/Users/zephyr/JupyterNotebook/NIRDiploma/DroneSemantic/SemanticDroneDataset/mask/',
            'class_mapping': {
            0: 0, 1: 3, 2: 1, 3: 1, 4: 1, 5: 2, 6: 0, 7: 0, 8: 0, 9: 0,
            10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 4, 16: 4, 17: 4, 18: 4,
            19: 0, 20: 0, 21: 5, 22: 0, 23: 0
            },
            'mask_type': 'grayscale'
        },
        {
            'name': 'swiss_okuna',
            'img_path': 'C:/Users/zephyr/JupyterNotebook/NIRDiploma/DroneSemantic/SwissOkuna/img',
            'mask_path': 'C:/Users/zephyr/JupyterNotebook/NIRDiploma/DroneSemantic/SwissOkuna/mask',
            'class_mapping': {0: 1, 1: 0, 2: 0, 3: 3, 4: 1, 5: 0, 6: 0, 7: 4, 8: 2, 9: 4},
            'mask_type': 'grayscale'
        }
    ]

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(num_classes).to(device)

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze decoder
    for param in model.decoder.parameters():
        param.requires_grad = True

    # Unfreeze segmentation head
    for param in model.segmentation_head.parameters():
        param.requires_grad = True

    # Optimizer, Scheduler, Scaler, and EarlyStopping
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), amsgrad=True, lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)
    scaler = GradScaler('cuda')

    # Load existing model (for evaluation or resuming training)
    if os.path.exists(model_path):
        print("Loading existing model")
        start_epoch, train_losses, val_losses, val_loss_min = load_checkpoint(model, optimizer, scheduler, scaler, model_path)
    else:
        print("No existing model found, starting from scratch")
        start_epoch, train_losses, val_losses, val_loss_min = 0, [], [], float('inf')

    early_stopping = EarlyStopping(patience=20, verbose=True, val_loss_min=val_loss_min)

    # Preprocess images (resize)
    image_resizer = ImageResizer(new_size=(W, H), num_workers=num_workers)
    image_resizer.preprocess_images(datasets_info)

    # Use the MultiDatasetLoader class to create combined loaders
    multi_loader = MultiDatasetLoader(datasets_info, batch_size, num_workers)
    combined_loaders = multi_loader.create_combined_loaders()

    # Extract combined loaders
    combined_train_loader = combined_loaders['train_loader']
    combined_valid_loader = combined_loaders['valid_loader']
    combined_test_loader = combined_loaders['test_loader']

    # Train the model on the combined dataset
    if args.mode == 'train':
        print('Training...')
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
        print('Evaluating...')
        evaluate_model(model, combined_test_loader, num_classes)
        plot_predictions(combined_test_loader, model, cmap, num_samples)

        if train_losses and val_losses:
            plot_loss(train_losses, val_losses)
        else:
            print("No loss data available to plot.")

    # Run the point-finding algorithm
    elif args.mode == 'points':
        print('Finding points...')
        model.eval()
        with torch.no_grad():
            data = next(iter(combined_test_loader))
            images = data[0]
            images = images.cuda()
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)

            plot_furthest_points(images.cpu(), preds.cpu(), cmap, zone_type, num_points, view_mode, num_samples)

    # Measure inference time without point-finding
    elif args.mode == 'inference':
        input_size = (1, 3, H, W)
        avg_time = measure_inference_time(model, device, input_size)
        print(f"Average inference time: {avg_time:.6f} seconds or {1/avg_time:.6f} FPS")

    gc_collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, Evaluate, Run Point-Finding Algorithm or Measure Inference Time")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'points', 'inference'], required=True, help="Choose whether to train, evaluate, run the point-finding algorithm or measure inference time")
    
    args = parser.parse_args()
    main(args)
