import segmentation_models_pytorch as smp
import os
import torch

def create_model(num_classes):
    model = smp.FPN(
        encoder_name="timm-mobilenetv3_large_minimal_100",       
        encoder_weights="imagenet",        
        in_channels=3,                     
        classes=num_classes                
    )
    return model

def initialize_model(config, device):
    """
    Initialize the model and move it to the correct device.
    """
    model = create_model(config['num_classes']).to(device)

    # Freeze encoder parameters if necessary
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze decoder and segmentation head
    for param in model.decoder.parameters():
        param.requires_grad = True

    for param in model.segmentation_head.parameters():
        param.requires_grad = True

    return model

def load_checkpoint(model, optimizer, scheduler, scaler, filename='checkpoint.pt'):
    """
    Load a checkpoint from a file and restore the model, optimizer, scheduler, and scaler.

    Args:
        model (torch.nn.Module): The model to be restored.
        optimizer (torch.optim.Optimizer): The optimizer to be restored.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to be restored.
        scaler (torch.cuda.amp.GradScaler): The gradient scaler to be restored.
        filename (str, optional): The name of the file to load the checkpoint from. Defaults to 'checkpoint.pt'.

    Returns:
        tuple: A tuple containing the epoch, training losses, validation losses, and the minimum validation loss.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    if val_losses:
        val_loss_min = min(val_losses)
    else:
        print("Warning: val_losses is empty, setting val_loss_min to infinity")
        val_loss_min = float('inf')

    print(f"Loaded checkpoint: Epoch: {epoch}, Validation Loss: {val_loss_min:.6f}")
    return epoch, train_losses, val_losses, val_loss_min

def load_existing_model(config, model, optimizer, scheduler, scaler):
    """
    Load the model state from a checkpoint if it exists.
    """
    if os.path.exists(config.get('model_path', '')):
        model_path = config.get('model_path')
    elif os.path.exists('models/checkpoint.pt'):
        model_path = 'models/checkpoint.pt'
    elif os.path.exists('models/final_model.pt'):
        model_path = 'models/final_model.pt'
    else:
        print("No existing model found, starting from scratch")
        return 0, [], [], float('inf')

    print(f"Loading existing model found in {model_path}")
    return load_checkpoint(model, optimizer, scheduler, scaler, model_path)
