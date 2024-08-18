# Landing zone semantic segmentation and finding optimal landing points for drone images
This repository contains code for a semantic segmentation project using deep learning models and algorithm for finding optimal landing points on segmented regions.

Includes: **.ipynb** and **.py** files.

**Stack:** Python, PyTorch, OpenCV, Albumentations, Segmentation Models PyTorch, scikit-learn, Matplotlib, tqdm, NumPy, SciPy

## Project Overview
The goal of this project is to train a semantic segmentation model that can classify different regions in drone-captured images for uni diploma. 

The model identifies and segments areas like roads, water bodies, landing zones, moving objects, and more.

Also there is algorithm for finding points for landing based on euclidean distance transform (distance_transform_edt). After it calculates score for optimal landing point on these criteria with weights:
- Distance from the center of the image or from the bottom part of the image (0.4)
- Zone size (0.2)
- Distance from zone borders (0.4)

## Features

- **DeepLabV3+ Model:** Utilizes the DeepLabV3+ architecture with MobileNetV2 as the backbone.
- **Image Preprocessing:** OpenCV and Albumentations for image resizing, augmentation, and normalization.
- **Custom Training Loop:** Includes a custom training loop with mixed precision training using PyTorch's AMP.
- **Early Stopping:** Implements early stopping to prevent overfitting during training.
- **Point Finding Algorithm:** Includes a post-processing step to identify the optimal points in segmented regions.

### Stack

`Python`, `PyTorch`, `OpenCV`, `Albumentations`, `Segmentation Models PyTorch`, `scikit-learn`, `Matplotlib`, `tqdm`, `NumPy`, `SciPy`, `Git`

### Setup Instructions on Windows

1. Clone the repository:
   ```bash
   git clone https://github.com/Chirhi/Optimal-Landing-Zone
   
2. Install the required packages:
    ```bash
    pip install -r requirements.txt

3. Prepare your dataset and adjust the paths in main.py accordingly.
   
4. Train the model:
    ```bash
    python main.py --mode train

5. Evaluate the model:
    ```bash
    python main.py --mode evaluate

6. Run the point-finding algorithm:
    ```bash
    python main.py --mode points

### Dataset
The dataset used for this project is **"Semantic Drone Dataset"** and it consists of 400 drone images and their corresponding segmentation masks.

URL: https://datasetninja.com/semantic-drone

### Model
The model uses a pretrained DeepLabV3+ architecture with a MobileNetV2 backbone. It is trained to segment drone imagery into several classes, including roads, water, landing zones, markers, moving objects and obstacles.

### Results

![Figure_1](https://github.com/user-attachments/assets/d540e55d-c8f4-403a-92fd-80bb4fe96835)

- Mean Precision: 0.8991
- Mean Recall: 0.8979
- Mean F1-Score: 0.8984
- Mean MIoU: 0.8011
- Class 0: Precision: 0.8750, Recall: 0.8950, F1-Score: 0.8849
- Class 1: Precision: 0.9378, Recall: 0.9189, F1-Score: 0.9282
- Class 2: Precision: 0.9382, Recall: 0.9522, F1-Score: 0.9452
- Class 3: Precision: 0.9579, Recall: 0.9573, F1-Score: 0.9576
- Class 4: Precision: 0.8400, Recall: 0.8485, F1-Score: 0.8442
- Class 5: Precision: 0.8460, Recall: 0.8152, F1-Score: 0.8303
