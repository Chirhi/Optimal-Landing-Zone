import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import torch
from utils import denormalize
import matplotlib.patches as mpatches

def filter_small_zones(mask, min_size=100):
    """
    Filters a binary mask by removing connected components smaller than a specified minimum size.

    Parameters:
        mask (numpy.ndarray): A binary mask where 0 represents the background and 1 represents the foreground.
        min_size (int, optional): The minimum size of a connected component to be kept in the mask. Defaults to 100.

    Returns:
        numpy.ndarray: A filtered binary mask where connected components smaller than min_size have been removed.
    """
    num_labels, labels_im = cv2.connectedComponents(mask.astype(np.uint8))
    filtered_mask = np.zeros_like(mask)
    for label in range(1, num_labels):
        if np.sum(labels_im == label) > min_size:
            filtered_mask[labels_im == label] = 1
    return filtered_mask

def find_multiple_furthest_points(mask, num_points):
    """
    Finds multiple furthest points from the edges of a given mask.

    Parameters:
    mask (numpy array): A 2D binary mask where 0 represents the edges and 1 represents the zone.
    num_points (int): The number of furthest points to find.

    Returns:
    list: A list of tuples containing the coordinates and maximum distances of the furthest points.
    """
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
    """
    Extend the mask to increase the boundaries of the objects.

    Parameters:
    mask (numpy array): The input mask to be dilated.
    kernel_size (int): The size of the kernel to use for dilation. Defaults to 15.

    Returns:
    numpy array: The dilated mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    return dilated_mask

def get_zone_mask(pred, zone_type='marker', dilate_kernel_size=60):
    """
    Returns a zone mask based on the type of zone and the dilation of moving objects.

    Parameters:
    pred (numpy array): The prediction array to generate the zone mask from.
    zone_type (str): The type of zone to generate the mask for. Can be 'safe', 'road', 'water', 'roadandwater', or 'marker'. Defaults to 'marker'.
    dilate_kernel_size (int): The size of the kernel to use for dilating the moving objects mask. Defaults to 60.

    Returns:
    numpy array: The generated zone mask.
    """
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
    """
    Calculates a score for a given point based on its position within an image, 
    the size of the zone it belongs to, and its distance from the image edges.

    Parameters:
        point (tuple): The coordinates of the point (y, x).
        zone_size (int): The size of the zone the point belongs to.
        center_coords (tuple): The coordinates of the center of the image (y, x).
        img_shape (tuple): The shape of the image (height, width).
        dist_from_edge (int): The distance from the point to the nearest edge.
        view_mode (str): The camera view mode, either 'forward' or 'bottom'.

    Returns:
        float: A score representing the point's position and zone size.
    """
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

def process_images_for_points(model, test_loader, device, num_samples):
    """
    Processes images for finding points.
    
    Parameters:
        model: Model for getting predictions
        test_loader: Test data loader
        device: Device for calculations (CPU/GPU)
        num_samples: Number of samples to process
    
    Returns:
        tuple: (images, preds) - processed images and their predictions
    """
    model.eval()
    with torch.no_grad():
        # Getting a random batch of images
        random_indices = torch.randperm(len(test_loader.dataset))[:num_samples]
        images = []
        preds = []
        
        for idx in random_indices:
            img, _ = test_loader.dataset[idx]
            img = img.unsqueeze(0).to(device)
            output = model(img)
            _, pred = torch.max(output.data, 1)
            
            images.append(img)
            preds.append(pred)
        
        # Combining all images and predictions
        images = torch.cat(images, dim=0)
        preds = torch.cat(preds, dim=0)
        
        return images.cpu(), preds.cpu()

def plot_furthest_points(images, preds, cmap, zone_type='marker', num_points=30, view_mode='bottom', num_samples = 1):
    """
    Plots the furthest points in a given image based on the specified zone type.

    Parameters:
    images (list): A list of input images.
    preds (list): A list of predictions corresponding to the input images.
    cmap (object): A colormap object used for visualization.
    zone_type (str): The type of zone to find points in (e.g., 'road', 'water', etc.).
    num_points (int, optional): The number of points to find in the specified zone. Defaults to 30.
    view_mode (str, optional): The camera view mode (e.g., 'bottom', 'forward'). Defaults to 'bottom'.
    """
    # Mean and standard deviation for denormalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Class labels for legend
    class_labels = ['Obstacles', 'Landing Zones', 'Water', 'Roads', 'Moving Objects', 'Marker']
    legend_patches = [mpatches.Patch(color=np.array(color)/255.0, label=label)
                      for color, label in zip(cmap, class_labels)]

    # Creating legend for the image with points
    landing_points_legend = [
        mpatches.Patch(color='yellow', label='Landing points', alpha=0.5),
        mpatches.Patch(color='lime', label='Optimal point')
    ]

    with torch.no_grad():
        for i in range(num_samples):
            image = images[i]
            pred = preds[i]

            def process_zone(zone_type):
                zone_mask = get_zone_mask(pred, zone_type)
                zone_filtered = filter_small_zones(zone_mask)
                bordered_mask = np.pad(zone_filtered, pad_width=1, mode='constant', constant_values=0)
                center_coords = (image.shape[1] // 2, image.shape[2] // 2)
                img_shape = image.shape[1:]  # (H, W)

                all_furthest_points = []
                num_labels, labels_im = cv2.connectedComponents(bordered_mask)
                for label in range(1, num_labels):
                    mask = (labels_im == label).astype(np.uint8)
                    zone_size = np.sum(mask)  # Size of the zone in pixels
                    num_zone_points = min(zone_size // 1500, num_points)  # Number of points proportional to the zone size
                    furthest_points = find_multiple_furthest_points(mask, num_zone_points)
                    furthest_points = [((point[0][0] - 1, point[0][1] - 1), point[1], zone_size) # Deleting the frame from coordinates
                                       for point in furthest_points] 
                    all_furthest_points.extend(furthest_points)

                all_furthest_points.sort(key=lambda p: calculate_score(
                    p[0], p[2], center_coords, img_shape, p[1], view_mode))
                return all_furthest_points, zone_type

            # Try to find points in the specified zone first
            all_furthest_points, current_zone_type = process_zone(zone_type)

            # If not found, find 'safe' points instead
            if len(all_furthest_points) == 0:
                print(f"No points found for zone type '{zone_type}'. Switching to 'safe' zone.")
                all_furthest_points, current_zone_type = process_zone('safe')

            # If not found, find 'road' points instead
            if len(all_furthest_points) == 0:
                print(f"No points found for 'safe' zone. Switching to 'road' zone.")
                all_furthest_points, current_zone_type = process_zone('road')

            # If no points found for any zone type, skip this sample
            if len(all_furthest_points) == 0:
                print(f"No points found for any zone type. Skipping this sample.")
                continue

            print(f"Points found using zone type: {current_zone_type}")

            fig, ax = plt.subplots(1, 3, figsize=(20, 5))
            ax[0].imshow(denormalize(image, mean, std))

            # Creating a colored prediction image
            pred_rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
            for class_idx in range(len(cmap)):
                pred_rgb[pred.numpy() == class_idx] = cmap[class_idx]

            ax[1].imshow(pred_rgb)
            ax[2].imshow(denormalize(image, mean, std))

            for point, dist, _ in all_furthest_points[:-1]:
                ax[2].scatter(point[1], point[0], c='yellow', s=20, alpha=0.6)

            # Highlighting the best point
            if len(all_furthest_points) > 0:
                best_point, best_dist, _ = all_furthest_points[-1]
                ax[2].scatter(best_point[1], best_point[0], c='lime', s=50, alpha=0.7)

            # Adding legend for prediction
            ax[1].legend(handles=legend_patches, loc='upper right', framealpha=0.3, fontsize='x-small', markerscale=0.7)
        
            # Adding legend for the image with points
            ax[2].legend(handles=landing_points_legend, loc='upper right', framealpha=0.3, fontsize='x-small', markerscale=0.7)

            ax[0].set_title('Original')
            ax[1].set_title('Prediction')
            ax[2].set_title('Optimal Points')
            for a in ax:
                a.get_xaxis().set_visible(False)
                a.get_yaxis().set_visible(False)
            plt.show()
