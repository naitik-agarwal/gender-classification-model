Gender Classification using CNN
Overview

This project implements a Convolutional Neural Network (CNN) to classify facial images as Male or Female.
The model is trained from scratch using TensorFlow/Keras and achieves approximately 87–88% test accuracy on the current dataset.

The goal of this project is to understand and build an end-to-end deep learning pipeline including:

Image preprocessing

CNN architecture design

Model training and evaluation

Performance analysis using classification metrics

Dataset

The dataset consists of approximately 1,000 facial images divided into two classes:

Female

Male

The images are resized to 100 × 100 pixels and normalized before training.

Data split:

80% Training

20% Testing
Stratified splitting is used to maintain class balance.

Model Architecture

The CNN architecture is built using Keras Sequential API.

Architecture:

Conv2D (32 filters, 3×3) + ReLU

MaxPooling (2×2)

Batch Normalization

Conv2D (64 filters, 3×3) + ReLU

MaxPooling (2×2)

Batch Normalization

Conv2D (128 filters, 3×3) + ReLU

MaxPooling (2×2)

Batch Normalization

Flatten

Dense (128 units, ReLU)

Dropout (0.5)

Dense (2 units, Softmax)

Loss Function: Categorical Crossentropy
Optimizer: Adam
Regularization: Dropout + Early Stopping

The convolutional layers extract spatial features such as edges, textures, and facial patterns.
The dense layers learn high-level feature interactions to perform classification.

Training Strategy

Images are normalized to range [0,1].

EarlyStopping is used to prevent overfitting.

Model training typically converges within 15–20 epochs.

Training accuracy approaches ~99%, while test accuracy stabilizes around 87–88%, indicating mild but controlled overfitting.

Evaluation

Final Test Results:

Test Accuracy: ~87–88%

Balanced precision and recall across both classes

Confusion matrix used to analyze class-wise performance

Example performance:

Female recall: ~84%

Male recall: ~92%

This shows the model generalizes reasonably well across both categories.

How to Run

Clone the repository

Create virtual environment:

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Run:

python GenderClassificationModel.py

Make sure the dataset folder structure is:

DATASET/
Female/
Male/
Challenges Faced

Initial class imbalance caused biased predictions.

Overfitting due to limited dataset size.

Sensitivity to lighting and facial orientation.

These were mitigated using:

Stratified train-test split

Dropout layers

Early stopping

Possible Improvements

Use RGB images instead of grayscale

Apply controlled data augmentation

Increase dataset size

Apply transfer learning (MobileNetV2 / EfficientNet)

Deploy as a web application (Flask/Streamlit)

What This Project Demonstrates

Understanding of CNN architecture

Data preprocessing pipeline

Model evaluation using classification metrics

Handling overfitting and class imbalance

Clean modular ML code structure
