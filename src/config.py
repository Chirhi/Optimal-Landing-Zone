import argparse
import glob
import os
import yaml

def filter_datasets(datasets):
    """
    Filters and removes empty or non-existent datasets from the configuration list.

    Args:
        datasets (list): List of datasets from the configuration, where each element is a dictionary with dataset parameters.

    Returns:
        list: Filtered list of datasets containing only valid datasets.
    """
    valid_datasets = []
    for dataset_info in datasets:
        img_path = dataset_info['img_path']
        mask_path = dataset_info['mask_path']
        
        # Check if paths exist and contain files
        if os.path.exists(img_path) and os.path.exists(mask_path):
            img_files = glob.glob(os.path.join(img_path, '*.jpg')) + glob.glob(os.path.join(img_path, '*.png')) + glob.glob(os.path.join(img_path, '*.jpeg'))
            mask_files = glob.glob(os.path.join(mask_path, '*.jpg')) + glob.glob(os.path.join(mask_path, '*.png')) + glob.glob(os.path.join(mask_path, '*.jpeg'))

            if img_files and mask_files:
                valid_datasets.append(dataset_info)
            else:
                print(f"Skipping dataset '{dataset_info['name']}': No images or masks found.")
        else:
            print(f"Skipping dataset '{dataset_info['name']}': Path does not exist.")
    
    # Update configuration with only valid datasets
    return valid_datasets

def parse_arguments(config):
    """
    Parse command-line arguments and override config values based on mode.

    Args:
        config (dict): Configuration dictionary loaded from config.yaml.

    Returns:
        tuple: Parsed arguments and updated configuration.
    """
    parser = argparse.ArgumentParser(description="Train, Evaluate, Run Point-Finding Algorithm or Measure Inference Time")

    # Common arguments
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'points', 'inference'], required=True, help="Choose operation mode")
    parser.add_argument('--model_path', type=str, default=config.get('model_path'), help="Relative path to the trained model file")
    parser.add_argument('--num_epochs', type=int, default=config.get('num_epochs', 200), help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 25), help="Batch size")
    parser.add_argument('--early_stopping_patience', type=int, default=config.get('early_stopping_patience', 20), help="Patience for early stopping")
    parser.add_argument('--num_workers', type=int, default=config.get('num_workers', 4), help="Number of workers for data loading")

    # Arguments for mode 'points' and 'evaluate'
    parser.add_argument('--zone_type', type=str, default='marker', help="Zone type for point finding")
    parser.add_argument('--view_mode', type=str, default='bottom', help="Camera view mode")
    parser.add_argument('--num_points', type=int, default=30, help="Number of points to find")

    args = parser.parse_args()

    # Update config based on mode
    if args.mode == 'train':
        config['num_epochs'] = args.num_epochs
        config['batch_size'] = args.batch_size
        config['early_stopping_patience'] = args.early_stopping_patience
        config['num_workers'] = args.num_workers
    elif args.mode == 'points':
        config['zone_type'] = args.zone_type
        config['view_mode'] = args.view_mode
        config['num_points'] = args.num_points

    return args, config

def load_config(project_root):
    """
    Load the configuration from config.yaml.

    Args:
        project_root (str): Root directory of the project.

    Returns:
        dict: Updated configuration dictionary.
    """
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing YAML configuration: {exc}")
    
    # Correct paths in config.yaml
    config['model_path'] = os.path.join(project_root, config['model_path'])
    
    # Correct paths for each dataset
    for dataset in config['datasets']:
        dataset['img_path'] = os.path.join(project_root, dataset['img_path'])
        dataset['mask_path'] = os.path.join(project_root, dataset['mask_path'])

    return config