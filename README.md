# Liver Tumor Segmentation Project

This repository contains a deep learning pipeline for the automated segmentation of the liver and liver tumors from CT scans (using the first 5 part of LiTS Dataset). It includes scripts for training, model inference, and an interactive 3D volume viewer.

Here is link to the dataset: https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation

# Part 1: Project Summary
**1. Dataset**

The project is designed to perform multi-class segmentation on CT scans. The segmentation identifies three distinct classes:

    Class 0: Background

    Class 1: Liver

    Class 2: Tumor

The training script (livertumor_v4.py) is configured to work with .npy files extracted from the LiTS (Liver Tumor Segmentation Challenge) dataset. The preprocessing involves clipping Hounsfield Units (HU) to a range of [-100, 250] to enhance the contrast of abdominal organs.

**2. Model Architecture**

The project utilizes the Unet++ architecture, a powerful evolution of the standard UNet that uses nested and dense skip connections to reduce the semantic gap between the feature maps of the encoder and decoder sub-networks.

    Encoder: ResNet-34.

    Pre-trained Weights: The encoder is initialized with ImageNet weights to accelerate convergence.

    Input Channels: 1 (Grayscale CT slices).

    Output Classes: 3.

**3. Training Details & Loss Function**

To handle the significant class imbalance (where tumors occupy much less space than the background), a hybrid loss function is used:

    TotalLoss = DiceLoss + CrossEntropyLoss

The Cross-Entropy component is further weighted with specific penalties: [0.1 (Background), 1.0 (Liver), 5.0 (Tumor)], forcing the model to prioritize accurate tumor detection.

**4. Evaluation Results**

The model is evaluated using common medical imaging metrics: Intersection over Union (IoU) and Dice Score (F1-Score).

    Best Model Selection: The "Best Model" is saved based on a combined score of (0.5 * Val_IoU + 0.5 * Val_Dice).

# Part 2: User Guide - How to Use

To use this model on your own CT data or continue training, follow the configuration steps below.

**Notes: Prefer using Nvidia GPU (with CUDA) to enhance the predicting or training speed**

**1. Environment Setup**

Ensure you have the following libraries installed:
    
    pip install torch torchvision segmentation-models-pytorch nibabel numpy opencv-python matplotlib tqdm

**2. Inference & Visualization (Running main.py)**

To visualize predictions on a .nii volume file, modify the following variables in main.py:

    MODEL_PATH: Path to your trained .pth model file.

    VOLUME_FILE_PATH: Path to the CT volume (.nii) you want to predict.

    MASK_FILE_PATH: (Optional) Path to the ground truth segmentation. If empty, the viewer will only show the original slice and the prediction.

    ROT_K: Adjust this (1, 2, or 3) if your NIfTI data appears rotated.

Interactive Controls:

    Scroll / Slider: Move through different slices of the CT volume.

    't' key: Toggle the visibility of the Ground Truth overlay mask (if provided).

**3. Training/Fine-tuning (Running livertumor_v4.py)**

If you wish to train the model on your own dataset:

    Prepare Data: Save your slices and masks as .npy files.

    Update Paths: Update TRAIN_IMG_DIR, TRAIN_MASK_DIR, and ROOT_DIR to point to your data folders.

    Resume Training: If you have a checkpoint, set RESUME = True and provide the CHECKPOINT_PATH.

    Hardware: The script automatically detects and uses CUDA if available.
    
Notes: check file extract_dataset.py for extracting data from .nii file to .npy for preparing dataset before training.

**4. Integrating the Predictor (Using LiverTumorModel.py)**

For integration into other Python applications, use the LiverTumorModel class. It handles the internal preprocessing (clipping and normalization) and provides a predict() method that returns the segmentation mask and the maximum tumor probability found in that slice.