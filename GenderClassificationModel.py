# %%
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer

# %%
path = os.listdir('C:/Users/garvm/OneDrive/Desktop/genderdatas/train')
classes = {'Female' : 0 , 'Male' : 1}

# %%
X = []
Y = []
for cls in classes:
    subpath = 'C:/Users/garvm/OneDrive/Desktop/genderdatas/train/'+cls
    for j in os.listdir(subpath):
        img_path = os.path.join(subpath, j)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
        if img is not None:
            img = cv2.resize(img, (100, 100))
            img = img.reshape(100, 100, 1)
            X.append(img)
            Y.append(classes[cls])


# %%
X = np.array(X)
Y = np.array(Y)
print("Class distribution:", pd.Series(Y).value_counts())


# %%
X = X.astype('float32') / 255.0

# %%
lb = LabelBinarizer()
Y = lb.fit_transform(Y)
Y = to_categorical(Y)


# %%
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, random_state=10, test_size=0.20, stratify=Y)


# %%
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# %%
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(xtrain, ytrain,epochs=30,batch_size=32,validation_data=(xtest, ytest),callbacks=[early_stopping])

test_loss, test_accuracy = model.evaluate(xtest, ytest)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")



# %%
y_pred = model.predict(xtest)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(ytest, axis=1)

# %%
print("\nClassification Report:\n", classification_report(y_true, y_pred_classes))
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred_classes))

# %%
def predict_gender():
    testimgpath = "C:\\Users\\garvm\\OneDrive\\Desktop\\Pendrive data\\trip\\Himachal\\IMG20170603103433.jpg"
    # Read the image
    img = cv2.imread(testimgpath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found or invalid")
        return

    # Resize the image
    img = cv2.resize(img, (100, 100))

    # Reshape for model input
    img = img.reshape(1, 100, 100, 1) / 255.0

    # Make the prediction using your model
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    if predicted_class == 0:
        return 'Female'
    else:
        return 'Male'
print("Predicted gender:", predict_gender())        

# %%



