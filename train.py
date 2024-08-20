import torch
from torch.amp import autocast
from tqdm import tqdm

def train_model(model, train_loader, valid_loader, num_classes, num_epochs, start_epoch, train_losses, val_losses, val_loss_min, device, optimizer, scheduler, scaler, early_stopping, criterion):
    """
    Trains a given model using the provided training and validation data loaders.

    Parameters:
    model (nn.Module): The model to be trained.
    train_loader (DataLoader): The data loader for the training data.
    valid_loader (DataLoader): The data loader for the validation data.
    num_classes (int): The number of classes in the classification problem.
    num_epochs (int): The number of epochs to train the model for.
    start_epoch (int): The starting epoch number.
    train_losses (list): A list to store the training losses at each epoch.
    val_losses (list): A list to store the validation losses at each epoch.
    val_loss_min (float): The minimum validation loss.
    device (torch.device): The device to use for training (GPU or CPU).
    optimizer (torch.optim.Optimizer): The optimizer to use for training.
    scheduler (torch.optim.lr_scheduler): The learning rate scheduler to use.
    scaler (GradScaler): The gradient scaler to use for mixed precision training.
    early_stopping (EarlyStopping): The early stopping criterion to use.
    criterion (nn.Module): The loss function to use for training.

    Returns:
    None
    """
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                # Mixed precision training
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss = loss / 4
                scaler.scale(loss).backward()

                # Gradient accumulation
                if (i + 1) % 4 == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (i + 1))
                pbar.update(1)

        train_losses.append(running_loss / len(train_loader))

        # Validate the model
        val_loss = validate_model(model, valid_loader, criterion, device)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        early_stopping(val_loss, model, epoch, optimizer, scheduler, scaler, train_losses, val_losses)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    if not early_stopping.early_stop:
        print("Saving final model")
        torch.save({
            'epoch': num_epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, 'final_model.pt')

def validate_model(model, valid_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(valid_loader)
