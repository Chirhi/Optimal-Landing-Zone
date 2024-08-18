import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import torch

def filter_small_zones(mask, min_size=100):
    """Filtering masks by size bigger than min_size"""
    num_labels, labels_im = cv2.connectedComponents(mask.astype(np.uint8))
    filtered_mask = np.zeros_like(mask)
    for label in range(1, num_labels):
        if np.sum(labels_im == label) > min_size:
            filtered_mask[labels_im == label] = 1
    return filtered_mask

def add_border(mask):
    """Adding a border around the image"""
    return np.pad(mask, pad_width=1, mode='constant', constant_values=0)

def find_multiple_furthest_points(mask, num_points):
    """Finding multiple furthest points from the edges of the zone"""
    furthest_points = []
    mask_copy = mask.copy()
    
    for _ in range(num_points):
        dist_transform = distance_transform_edt(mask_copy)
        max_dist = np.max(dist_transform)
        if max_dist == 0:
            break  # Stopping if there are no more points to find
        
        max_dist_coords = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
        furthest_points.append((max_dist_coords, max_dist))
        
        # Marking the found point as a border to find the next most distant point
        mask_copy[max_dist_coords] = 0
        
    return furthest_points

def dilate_mask(mask, kernel_size=15):
    """Extend the mask to increase the boundaries of the objects"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    return dilated_mask

def get_zone_mask(pred, zone_type='marker', dilate_kernel_size=60):
    """Getting zone mask by the type of the zone and the dilation of moving objects"""
    moving_objects_mask = (pred == 4).numpy().astype(np.uint8)
    dilated_moving_objects_mask = dilate_mask(moving_objects_mask, kernel_size=dilate_kernel_size)
    
    if zone_type == 'safe':
        zone_mask = ((pred == 1) | (pred == 5)).numpy().astype(np.uint8)
    elif zone_type == 'road':
        zone_mask = ((pred == 1) | (pred == 3) | (pred == 5)).numpy().astype(np.uint8)
    elif zone_type == 'water':
        zone_mask = ((pred == 1) | (pred == 2) | (pred == 5)).numpy().astype(np.uint8)
    elif zone_type == 'roadandwater':
        zone_mask = ((pred == 1) | (pred == 2) | (pred == 3) | (pred == 5)).numpy().astype(np.uint8)
    elif zone_type == 'marker':
        zone_mask = (pred == 5).numpy().astype(np.uint8)
    else:
        raise ValueError(f"Unknown zone_type: {zone_type}")

    # Excluding dilated moving objects from the zone
    zone_mask[dilated_moving_objects_mask == 1] = 0
    return zone_mask

def calculate_score(point, zone_size, center_coords, img_shape, dist_from_edge, view_mode):
    """Calculation of the score of the point based on its position and the zone size"""
    y, x = point

    if view_mode == 'forward':
        bottom_center_coords = (img_shape[0] - 1, img_shape[1] // 2)
        distance_from_reference = np.sqrt((x - bottom_center_coords[1])**2 + (y - bottom_center_coords[0])**2)
        distance_from_reference_norm = distance_from_reference / np.sqrt(img_shape[0]**2 + img_shape[1]**2)
    elif view_mode == 'bottom':
        distance_from_reference = np.sqrt((x - center_coords[1])**2 + (y - center_coords[0])**2)
        distance_from_reference_norm = distance_from_reference / np.sqrt(img_shape[0]**2 + img_shape[1]**2)
    else:
        raise ValueError(f"Unknown camera view: {view_mode}")

    size_score = zone_size / (img_shape[0] * img_shape[1])
    dist_from_edge_norm = dist_from_edge / np.max([img_shape[0], img_shape[1]])

    # Distance from the center of the image or from the bottom part of the image, zone size and distance from image borders
    score = 0.4 * (1 - distance_from_reference_norm) + 0.2 * size_score + 0.4 * dist_from_edge_norm
    return score

def plot_furthest_points(images, masks, preds, cmap, zone_type, num_points=30, view_mode='bottom'):
    """
    Plots the furthest points in a given image based on the specified zone type.

    Parameters:
    images (list): A list of input images.
    masks (list): A list of masks corresponding to the input images.
    preds (list): A list of predictions corresponding to the input images.
    cmap (object): A colormap object used for visualization.
    zone_type (str): The type of zone to find points in (e.g., 'road', 'water', etc.).
    num_points (int, optional): The number of points to find in the specified zone. Defaults to 30.
    view_mode (str, optional): The camera view mode (e.g., 'bottom', 'forward'). Defaults to 'bottom'.

    Returns:
    None
    """
    with torch.no_grad():
        for i in range(1):  # Change to plot only one sample
            image = images[i]
            pred = preds[i]

            def process_zone(zone_type):
                zone_mask = get_zone_mask(pred, zone_type)
                zone_filtered = filter_small_zones(zone_mask)
                bordered_mask = add_border(zone_filtered)
                center_coords = (image.shape[1] // 2, image.shape[2] // 2)
                img_shape = image.shape[1:]  # (H, W)

                all_furthest_points = []
                num_labels, labels_im = cv2.connectedComponents(bordered_mask)
                for label in range(1, num_labels):
                    mask = (labels_im == label).astype(np.uint8)
                    zone_size = np.sum(mask)  # Size of the zone in pixels
                    num_zone_points = min(zone_size // 1500, num_points)  # Number of points proportional to the zone size
                    furthest_points = find_multiple_furthest_points(mask, num_zone_points)
                    # Deleting the frame from coordinates
                    furthest_points = [((point[0][0] - 1, point[0][1] - 1), point[1], zone_size) for point in furthest_points]
                    all_furthest_points.extend(furthest_points)

                all_furthest_points.sort(key=lambda p: calculate_score(p[0], p[2], center_coords, img_shape, p[1], view_mode))
                return all_furthest_points

            # Try to find points in the specified zone first
            all_furthest_points = process_zone(zone_type)

            # If not found, find 'safe' points instead
            if len(all_furthest_points) == 0:
                all_furthest_points = process_zone('safe')

            # If not found, find 'road' points instead
            if len(all_furthest_points) == 0:
                all_furthest_points = process_zone('road')

            fig, ax = plt.subplots(1, 3, figsize=(20, 5))
            ax[0].imshow(np.transpose(image.numpy(), (1, 2, 0)))
            ax[1].imshow(cmap[pred.numpy()])
            ax[2].imshow(np.transpose(image.numpy(), (1, 2, 0)))

            for point, dist, _ in all_furthest_points[:-1]:
                ax[2].scatter(point[1], point[0], c='yellow', s=20)

            # Выделение самой оптимальной точки
            if len(all_furthest_points) > 0:
                best_point, best_dist, _ = all_furthest_points[-1]
                ax[2].scatter(best_point[1], best_point[0], c='lime', s=50)

            ax[0].set_title('Original')
            ax[1].set_title('Prediction')
            ax[2].set_title('Optimal Points')
            for a in ax:
                a.get_xaxis().set_visible(False)
                a.get_yaxis().set_visible(False)
            plt.show()
