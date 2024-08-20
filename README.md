# Landing zone semantic segmentation and finding optimal landing points for drone images
This repository contains code for a semantic segmentation project using deep learning models and algorithm for finding optimal landing points on segmented regions.

Includes: **.ipynb** (old) and **.py** files.

**Stack:** Python, PyTorch, OpenCV, Albumentations, Segmentation Models PyTorch, scikit-learn, Matplotlib, tqdm, NumPy, SciPy

## Project Overview
The goal of this project is to train a semantic segmentation model that can classify different regions in drone-captured images for uni diploma. 

The model identifies and segments areas like roads, water, landing zones, markers, moving objects and obstacles.

Also there is algorithm for finding points for landing based on euclidean distance transform (distance_transform_edt). After it calculates score for optimal landing point on criteria with weights:
- Distance from the center of the image or from the bottom part of the image (0.4)
- Zone size (0.2)
- Distance from zone borders (0.4)

## Features

- **DeepLabV3+ Model:** Utilizes the DeepLabV3+ architecture with MobileNetV2 as the backbone.
- **Image Preprocessing:** OpenCV and Albumentations for image resizing, augmentation, and normalization.
- **Multiple datasets:** Supports including multiple datasets and remapping their classes.
- **Custom Training Loop:** Includes a custom training loop with mixed precision training, learning rate scheduler, early stopping, gradient accumulation.
- **Point Finding Algorithm:** Includes a post-processing step to identify the optimal points in segmented regions.

## Setup Instructions on Windows

1. Clone the repository:
   ```bash
   git clone https://github.com/Chirhi/Optimal-Landing-Zone
   
2. Install the required packages:
    ```bash
    pip install -r requirements.txt

3. Prepare your dataset and adjust the paths and classes in main.py accordingly.
   
4. Train the model:
    ```bash
    python main.py --mode train

5. Evaluate the model:
    ```bash
    python main.py --mode evaluate

6. Run the point-finding algorithm:
    ```bash
    python main.py --mode points

## Datasets
The datasets used for this project are:
- **"Semantic Drone Dataset"**. Consists of 400 aerial drone images.
URL: https://datasetninja.com/semantic-drone.
- **Swiss Drone and Okutama Drone Datasets**. Consists of 191 aerial drone images.
URL: https://www.kaggle.com/datasets/aletbm/swiss-drone-and-okutama-drone-datasets

## Results

![Figure_1](https://github.com/user-attachments/assets/8940eaa7-6166-4e02-a1bc-7ecc1d3143b3)

- Loaded checkpoint: Epoch: 72, Validation Loss: 0.236323
- Mean Recall: 0.8825
- Mean F1-Score: 0.8766
- Mean MIoU: 0.7679
- Class 0: Precision: 0.8829, Recall: 0.9331, F1-Score: 0.9073
- Class 1: Precision: 0.9448, Recall: 0.8955, F1-Score: 0.9195
- Class 2: Precision: 0.8148, Recall: 0.9445, F1-Score: 0.8748
- Class 3: Precision: 0.9492, Recall: 0.9441, F1-Score: 0.9466
- Class 4: Precision: 0.7875, Recall: 0.6916, F1-Score: 0.7364
- Class 5: Precision: 0.8636, Recall: 0.8860, F1-Score: 0.8747
