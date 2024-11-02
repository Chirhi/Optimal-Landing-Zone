import numpy as np
import torch
import gc
import matplotlib.pyplot as plt
import time

def remap_labels(mask, class_mapping, placeholder_class=None):
    """
    Remaps the labels in a given mask using a provided class mapping.

    Args:
        mask (numpy.ndarray): The input mask to be remapped.
        class_mapping (dict): A dictionary mapping old class labels to new class labels.

    Returns:
        numpy.ndarray: The remapped mask with the labels replaced according to the class mapping.
    """
    new_mask = np.zeros_like(mask)
    for old_class, new_class in class_mapping.items():
        if old_class == "placeholder_class" and placeholder_class is not None:
            new_class = placeholder_class
        new_mask[mask == old_class] = new_class
    return new_mask

def denormalize(image, mean, std):
    """
    Denormalizes an image by reversing the normalization process in transforms module.

    Args:
        image (torch.Tensor): The normalized image tensor.
        mean (list or tuple): The mean values used for normalization.
        std (list or tuple): The standard deviation values used for normalization.

    Returns:
        np.ndarray: The denormalized image.
    """
    mean = np.array(mean)
    std = np.array(std)
    image = image.permute(1, 2, 0).numpy()
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image

def plot_loss(train_losses, val_losses):
    """
    Plots the training and validation loss over epochs.

    Args:
        train_losses (list): A list of training losses at each epoch.
        val_losses (list): A list of validation losses at each epoch.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()


def measure_inference_time(model, device, input_size=(1, 3, 224, 224), num_iterations=100):
    """
    Measures the average inference time of a given model on a specified device.

    Args:
        model: The model to measure inference time for.
        device: The device to run the model on.
        input_size (tuple, optional): The size of the input tensor. Defaults to (1, 3, 224, 224).
        num_iterations (int, optional): The number of iterations to run the model for. Defaults to 100.
        find_points_func (function, optional): The function for finding points. Should accept images and predictions.

    Returns:
        float: The average inference time of the model.
    """

    # Create dummy input
    dummy_input = torch.randn(input_size).to(device)
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)
    
    # Measure inference time
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            model(dummy_input)
    
    end_time = time.perf_counter()
    
    avg_inference_time = (end_time - start_time) / num_iterations
    
    return avg_inference_time
