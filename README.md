# Landing zone semantic segmentation and finding optimal landing points for drone images
This repository contains code for a semantic segmentation project using deep learning models and algorithm for finding optimal landing points on segmented regions.

**Stack:** Python, PyTorch, OpenCV, Albumentations, Segmentation Models PyTorch, scikit-learn, Matplotlib, tqdm, NumPy, SciPy

## Project Overview
The objective of this project is to develop a semantic segmentation model capable of accurately classifying various regions in images captured by drones, as part of a final qualifying project at the university. 

The model will be designed to detect and segment areas such as roads, bodies of water, landing zones, markers, moving objects, and obstacles.

In addition, the project includes an algorithm for identifying optimal landing points using the Euclidean distance transform (distance_transform_edt). The algorithm evaluates potential landing sites based on a weighted scoring system, with criteria including:

- Proximity to the center or bottom part of the image (weight: 0.4)
- Size of the landing zone (weight: 0.2)
- Distance from the borders of the landing zone (weight: 0.4)

This approach ensures that the chosen landing point is both safe and strategically located based on the defined criteria.

## Features

- **FPN and MobileNetV3 Model:** Utilizes the FPN architecture with MobileNetV3-Large-Minimal as the backbone for fast inference and optimal perfomance.
- **Image Preprocessing:** Albumentations for image resizing, augmentation, and normalization.
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

3. Prepare your dataset locally and adjust the paths and variables in config.yaml accordingly. Datasets should contain img and grayscale mask folders without subfolders. Algorithm will create resized images folders and subfolders for training, validation and testing itself after. For example:
   ```bash
   datasets
   ├───SemanticDroneDataset
   │   ├───img
   │   └───mask
   └───SwissOkuna
       ├───img
       └───mask
   
4. Train the model:
    ```bash
    python src/main.py --mode train

5. Evaluate the model: (may not work atm)
    ```bash
    python src/main.py --mode evaluate

6. Run the point-finding algorithm:
    ```bash
    python src/main.py --mode points

7. Run inference measurument:
    ```bash
    python src/main.py --mode inference

8. Also you can use console arguments for every mode if needed. Default values:
   ```bash
   --model_path = models/final_model.pt
   --num_epochs = 200
   --batch_size = 10
   --early_stopping_patience = 20
   --num_workers = 4
   --zone_type = marker
   --view_mode = bottom
   --num_points = 30

## Datasets
The datasets used for this project are:
- **"Semantic Drone Dataset"**. Consists of 400 aerial drone images.
URL: https://datasetninja.com/semantic-drone.
- **Swiss Drone and Okutama Drone Datasets**. Consists of 191 aerial drone images.
URL: https://www.kaggle.com/datasets/aletbm/swiss-drone-and-okutama-drone-datasets
- **Aerial Semantic Segmentation (Aeroscapes)**. Consists of 3269 aerial drone images. (need reviewing class mapping)
URL: https://www.kaggle.com/datasets/kooaslansefat/uav-segmentation-aeroscapes

## Results
![Figure_1](https://github.com/user-attachments/assets/70f27202-4cc8-4397-8c6f-7e165a4f6799)
![Figure_2](https://github.com/user-attachments/assets/f74f4ab0-cc5d-4c83-b922-d5c1e9f6cfb7)

- Loaded checkpoint: Epoch: 210, Validation Loss: 0.295837
- Mean Precision: 0.8845
- Mean Recall: 0.8473
- Mean F1-Score: 0.8647
- Mean MIoU: 0.7066
- Class 0: Precision: 0.8709, Recall: 0.8865, F1-Score: 0.8786
- Class 1: Precision: 0.8833, Recall: 0.8943, F1-Score: 0.8888
- Class 2: Precision: 0.9023, Recall: 0.8804, F1-Score: 0.8912
- Class 3: Precision: 0.9215, Recall: 0.9045, F1-Score: 0.9130
- Class 4: Precision: 0.8228, Recall: 0.7236, F1-Score: 0.7700
- Class 5: Precision: 0.9060, Recall: 0.7946, F1-Score: 0.8467
