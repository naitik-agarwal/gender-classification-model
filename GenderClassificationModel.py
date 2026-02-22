import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical



# CONFIG

DATA_DIR = "DATASET"       
IMG_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 30
CLASSES = {"Female": 0, "Male": 1}



# DATA LOADING

def load_data():
    X, Y = [], []

    for cls in CLASSES:
        subpath = os.path.join(DATA_DIR, cls)
        for file in os.listdir(subpath):
            img_path = os.path.join(subpath, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.reshape(IMG_SIZE, IMG_SIZE, 1)
                X.append(img)
                Y.append(CLASSES[cls])

    X = np.array(X).astype("float32") / 255.0
    Y = to_categorical(np.array(Y), num_classes=2)

    print("Class distribution:")
    print(pd.Series(np.argmax(Y, axis=1)).value_counts())

    return train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)



# MODEL

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model



# TRAINING

def train_model(model, xtrain, ytrain, xtest, ytest):
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        xtrain, ytrain,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(xtest, ytest),
        callbacks=[early_stopping]
    )

    return history



# EVALUATION

def evaluate_model(model, xtest, ytest):
    loss, accuracy = model.evaluate(xtest, ytest)
    print(f"\nTest Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    y_pred = model.predict(xtest)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(ytest, axis=1)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred_classes))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred_classes))



# SINGLE IMAGE PREDICTION

def predict_image(model, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    return "Female" if predicted_class == 0 else "Male"



# MAIN

if __name__ == "__main__":
    xtrain, xtest, ytrain, ytest = load_data()
    model = build_model()
    train_model(model, xtrain, ytrain, xtest, ytest)
    evaluate_model(model, xtest, ytest)

    # Example prediction
    sample_image = "sample.jpg"   # put any test image inside project folder
    if os.path.exists(sample_image):
        print("Predicted gender:", predict_image(model, sample_image))