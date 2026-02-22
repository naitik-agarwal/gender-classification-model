Gender Classification using CNN
Overview

This project builds a Convolutional Neural Network (CNN) to classify facial images as Male or Female.

The model is trained from scratch using TensorFlow/Keras and achieves ~87–88% test accuracy.

It demonstrates a complete deep learning pipeline:
Image preprocessing
CNN architecture design
Model training and evaluation
Performance analysis using classification metrics

Dataset
~1,000 facial images

Two classes: Female and Male

Images resized to 100 × 100

Pixel values normalized to [0,1]

80% training / 20% testing (stratified split)

Folder structure:
DATASET/
Female/
Male/
Model Architecture

Built using Keras Sequential API.

Architecture:

Conv2D (32) → MaxPool → BatchNorm
Conv2D (64) → MaxPool → BatchNorm
Conv2D (128) → MaxPool → BatchNorm
Flatten
Dense (128, ReLU)
Dropout (0.5)
Dense (2, Softmax)

Training Setup

Loss: Categorical Crossentropy

Optimizer: Adam

Regularization: Dropout + Early Stopping

The convolution layers extract spatial features (edges, textures, patterns), while dense layers learn high-level representations for classification.

Results

Test Accuracy: ~87–88%
Female recall: ~84%
Male recall: ~92%

The model generalizes well with mild but controlled overfitting (training accuracy ~99%).

How to Run

Clone the repository and create a virtual environment:

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Run:
python GenderClassificationModel.py

Challenges
Class imbalance in early experiments
Overfitting due to limited dataset size
Sensitivity to lighting and pose variations

Mitigation techniques:
Stratified splitting
Dropout
Early stopping

Future Improvements
Use RGB instead of grayscale
Apply data augmentation
Increase dataset size
Use transfer learning (MobileNet / EfficientNet)
Deploy as a web application

What This Project Shows
Understanding of CNN fundamentals
Clean data preprocessing pipeline
Proper evaluation using precision/recall/F1-score
Handling overfitting and imbalance
Modular ML project structure
