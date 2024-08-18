import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def calculate_miou(model, loader, num_classes):
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

def calculate_iou(preds, labels, num_classes):
    iou_list = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (labels == cls)).sum().item()
        union = ((preds == cls) | (labels == cls)).sum().item()
        if union == 0:
            iou_list.append(np.nan)
        else:
            iou_list.append(intersection / union)
    return np.nanmean(iou_list)

def evaluate_model(model, loader, num_classes):
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
    model.eval()
    with torch.no_grad():
        data = next(iter(loader))
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        _, predictions = torch.max(outputs.data, 1)
        images, labels, predictions = images.cpu(), labels.cpu(), predictions.cpu()

        for i in range(num_samples):
            fig, ax = plt.subplots(1, 3, figsize=(20, 5))
            ax[0].imshow(np.transpose(images[i].numpy(), (1, 2, 0)))  # Оригинальное изображение
            ax[1].imshow(cmap[labels[i].numpy()])  # Метки
            ax[2].imshow(cmap[predictions[i].numpy()])  # Предсказание
            ax[0].set_title('Оригинал')
            ax[1].set_title('Маска')
            ax[2].set_title('Предсказание')
            for a in ax:
                a.get_xaxis().set_visible(False)
                a.get_yaxis().set_visible(False)
            plt.show()

def calculate_precision_recall_f1(model, loader, num_classes):
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
