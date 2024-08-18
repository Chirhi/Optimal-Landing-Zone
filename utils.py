import numpy as np
import torch
import gc

def remap_labels(mask, class_mapping):
    new_mask = np.zeros_like(mask)
    for old_class, new_class in class_mapping.items():
        new_mask[mask == old_class] = new_class
    return new_mask

def load_checkpoint(model, optimizer, scheduler, scaler, filename='checkpoint.pt'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    # train_mious = checkpoint['train_mious']
    # val_mious = checkpoint['val_mious']
    val_loss_min = min(val_losses) if val_losses else np.Inf
    return epoch, train_losses, val_losses, val_loss_min

def plot_learning_curves(train_losses, val_losses):
    import matplotlib.pyplot as plt
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_losses, 'b', label='Train Loss')
    plt.plot(epochs, val_losses, 'r', label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()
