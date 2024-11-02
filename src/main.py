import os
import torch
import sys
from pathlib import Path
import torch.optim as optim
import random
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler
from train import train_model
from eval import evaluate_model, plot_predictions
from model import initialize_model, load_existing_model
from utils import (
    gc_collect, 
    plot_loss, 
    measure_inference_time
)
from image_resizer import ImageResizer
from multidataset_loader import MultiDatasetLoader
from config import parse_arguments, load_config, filter_datasets
from early_stopping import EarlyStopping
from optimal_point import plot_furthest_points, process_images_for_points

def main():
    """
    This is the main function that controls the entire program flow. It takes in a single argument 'args' which is expected to be a namespace object containing the mode of operation.

    The function supports three modes of operation:
    - 'train': This mode trains a deep learning model on a combined dataset of images and their corresponding masks.
    - 'evaluate': This mode evaluates the performance of a trained model on a test dataset and plots the results.
    - 'points': This mode runs a point-finding algorithm on a test dataset using a trained model.
    - 'inference': This mode measures the inference time of a trained model.
    """

    # os.environ['TORCH_USE_CUDA_DSA'] = '1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # print("TORCH_USE_CUDA_DSA:", os.getenv("TORCH_USE_CUDA_DSA"))
    # print("CUDA_LAUNCH_BLOCKING:", os.getenv("CUDA_LAUNCH_BLOCKING"))
    
    # Define the project root directory
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.append(str(PROJECT_ROOT / 'src'))

    # Load configuration
    config = load_config(PROJECT_ROOT)

    # Set random seeds
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # Filter datasets for empty datasets
    config['datasets'] = filter_datasets(config['datasets'])

    # Parse command-line arguments
    args, config = parse_arguments(config)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = initialize_model(config, device)

    # Initialize optimizer, scheduler, and early stopping
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        amsgrad=True,
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['factor'],
        patience=config['patience'],
        min_lr=config['min_lr']
    )
    scaler = GradScaler('cuda')

    # Load existing model (for evaluation or resuming training)
    start_epoch, train_losses, val_losses, val_loss_min = load_existing_model(
        config, model, optimizer, scheduler, scaler
    )

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'], 
        verbose=True, 
        val_loss_min=val_loss_min
    )

    # Preprocess images
    image_resizer = ImageResizer(
        new_size=(config['image_dimensions'][1], config['image_dimensions'][0]), 
        num_workers=config['num_workers']
    )
    image_resizer.preprocess_images(config['datasets'])

    # Use MultiDatasetLoader to create loaders
    multi_loader = MultiDatasetLoader(
        config['datasets'], 
        config['batch_size'], 
        config['num_workers'],
        config['image_dimensions']
    )
    combined_loaders = multi_loader.create_combined_loaders()

    # Extract combined loaders
    combined_train_loader = combined_loaders['train_loader']
    combined_valid_loader = combined_loaders['valid_loader']
    combined_test_loader = combined_loaders['test_loader']

    # Main workflow based on mode
    if args.mode == 'train':
        print(f"-------------------------------- \n Starting training...")
        train_model(
            model, 
            combined_train_loader, 
            combined_valid_loader, 
            config['num_classes'], 
            config['num_epochs'], 
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

    # Evaluate the model and show only predictions
    elif args.mode == 'evaluate':
        print('Evaluating model...')
        evaluate_model(model, combined_test_loader, config['num_classes'])
        plot_predictions(combined_test_loader, model, config.get('cmap'), config.get('num_samples', 1))

        if train_losses and val_losses:
            plot_loss(train_losses, val_losses)
        else:
            print("No loss data available to plot.")

    # Find points based on model segmentation masks
    elif args.mode == 'points':
        print('Running point-finding algorithm...')
        images, preds = process_images_for_points(
            model, 
            combined_test_loader, 
            device, 
            config.get('num_samples', 1)
        )
        plot_furthest_points(
            images, preds, config.get('cmap'),
            config['zone_type'], config['num_points'],
            config['view_mode'], config.get('num_samples', 1)
        )
    
    # Measure inference time without point-finding
    elif args.mode == 'inference':
        print('Measuring inference time...')
        input_size = (1, 3, config['image_dimensions'][0], config['image_dimensions'][1])
        avg_time = measure_inference_time(model, device, input_size)
        print(f"Average inference time: {avg_time:.6f} seconds or {1/avg_time:.6f} FPS")

    gc_collect()

if __name__ == "__main__":
    main()
