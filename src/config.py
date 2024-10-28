import argparse
import os
import yaml

def parse_arguments(config):
    """
    Parse command-line arguments and override config values based on mode.
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