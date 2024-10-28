import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from utils import denormalize

def calculate_precision_recall_f1(model, loader, num_classes):
    """
    Calculates the precision, recall, and F1 score of a given model on a dataset.

    Parameters:
        model: The model to be evaluated.
        loader: The data loader containing the images and labels.
        num_classes: The number of classes in the classification problem.

    Returns:
        precision: The precision score of the model, averaged over all classes.
        recall: The recall score of the model, averaged over all classes.
        f1: The F1 score of the model, averaged over all classes.
    """
    model.eval()
    preds_list = []
    labels_list = []
    
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            preds_list.append(preds.cpu().numpy().flatten())
            labels_list.append(labels.cpu().numpy().flatten())
    
    preds_array = np.concatenate(preds_list)
    labels_array = np.concatenate(labels_list)
    
    precision = precision_score(labels_array, preds_array, average=None, labels=range(num_classes))
    recall = recall_score(labels_array, preds_array, average=None, labels=range(num_classes))
    f1 = f1_score(labels_array, preds_array, average=None, labels=range(num_classes))
    
    return precision, recall, f1

def calculate_iou(preds, labels, num_classes):
    """
    Calculates the Intersection over Union (IoU) score for each class in the given predictions and labels.

    Parameters:
        preds (numpy.ndarray): The predicted labels.
        labels (numpy.ndarray): The ground truth labels.
        num_classes (int): The number of classes in the dataset.

    Returns:
        float: The mean IoU score across all classes.
    """
    iou_list = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (labels == cls)).sum().item()
        union = ((preds == cls) | (labels == cls)).sum().item()
        if union == 0:
            iou_list.append(np.nan)
        else:
            iou_list.append(intersection / union)
    return np.nanmean(iou_list)

def calculate_miou(model, loader, num_classes):
    """
    Calculates the mean Intersection over Union (mIoU) score for a given model, data loader, and number of classes.

    Args:
        model: The model to evaluate.
        loader: The data loader containing the images and labels.
        num_classes: The number of classes in the dataset.

    Returns:
        The mean IoU score.
    """
    model.eval()
    ious = []
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            iou = calculate_iou(preds.cpu().numpy(), labels.cpu().numpy(), num_classes)
            ious.append(iou)
    mean_iou = np.nanmean(ious)
    return mean_iou

def evaluate_model(model, loader, num_classes):
    """
    Evaluates a given model on a dataset.

    Parameters:
        model: The model to be evaluated.
        loader: The data loader containing the images and labels.
        num_classes: The number of classes in the dataset.
    """
    # Calculate Precision, Recall, and F1-Score
    precision, recall, f1 = calculate_precision_recall_f1(model, loader, num_classes)
    miou = calculate_miou(model, loader, num_classes)
    
    print(f'Mean Precision: {np.mean(precision):.4f}')
    print(f'Mean Recall: {np.mean(recall):.4f}')
    print(f'Mean F1-Score: {np.mean(f1):.4f}')
    print(f'Mean MIoU: {miou:.4f}')
    
    for cls in range(num_classes):
        print(f'Class {cls}: Precision: {precision[cls]:.4f}, Recall: {recall[cls]:.4f}, F1-Score: {f1[cls]:.4f}')

def plot_predictions(loader, model, cmap, num_samples=5):
    """
    Plots the predictions of a given model on a dataset.

    Parameters:
        loader: The data loader containing the images and labels.
        model: The model to be evaluated.
        cmap: The color map to be used for visualization.
        num_samples: The number of samples to be plotted (default is 5).
    """
    model.eval()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    with torch.no_grad():
        data = next(iter(loader))
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        _, predictions = torch.max(outputs.data, 1)
        images, labels, predictions = images.cpu(), labels.cpu(), predictions.cpu()

        for i in range(num_samples):
            fig, ax = plt.subplots(1, 3, figsize=(20, 5))
            ax[0].imshow(denormalize(images[i], mean, std))
            ax[1].imshow(cmap[labels[i].numpy()])
            ax[2].imshow(cmap[predictions[i].numpy()])
            ax[0].set_title('Original')
            ax[1].set_title('Mask')
            ax[2].set_title('Prediction')
            for a in ax:
                a.get_xaxis().set_visible(False)
                a.get_yaxis().set_visible(False)
            plt.show()
