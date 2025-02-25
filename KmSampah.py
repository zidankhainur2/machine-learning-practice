#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gdown
import os
import zipfile
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

# ==============================
# 📁 Download & Extract Dataset
# ==============================
url = 'https://drive.google.com/file/d/1wyvrU033jB81YeOPymxd_B8oI4eXLjb6/view?usp=drive_link'
output = 'garbage_classification.zip'

gdown.download(url, output, quiet=False)

if os.path.exists(output):
    try:
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall('garbage_classification')
        print("File berhasil diekstrak.")
    except zipfile.BadZipFile:
        print("File tidak valid atau bukan file ZIP.")
else:
    print("File tidak ditemukan atau tidak valid.")

# ==============================
# 📊 Data Preprocessing
# ==============================
dataset_path = 'garbage_classification'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, 'train'),
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    os.path.join(dataset_path, 'validation'),
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# ==============================
# 🤖 Model Building (MobileNetV2)
# ==============================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # Freeze pretrained layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(6, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ==============================
# 🧠 Callbacks
# ==============================
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# ==============================
# 📈 Model Training
# ==============================
history = model.fit(
    train_generator,
    epochs=1,
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# ==============================
# 📊 Evaluate & Plot Results
# ==============================
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy*100:.2f}%")

def plot_learning_curve(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

plot_learning_curve(history)


# In[4]:


# ==============================
# 🖼️ Image Prediction Function
# ==============================
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

model = load_model('best_model.keras')

def detect_and_classify_trash(image):
    orig_image = image.copy()
    resized_image = cv2.resize(image, (150, 150))
    normalized_image = resized_image.astype("float") / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)

    predictions = model.predict(input_image)
    predicted_class_indices = np.argmax(predictions, axis=1)
    class_names = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]

    class_predictions = {class_names[i]: predictions[0][i] for i in range(len(class_names))}
    sorted_predictions = sorted(class_predictions.items(), key=lambda x: x[1], reverse=True)

    # Pie Chart
    labels = [class_name for class_name, _ in sorted_predictions]
    scores = [score for _, score in sorted_predictions]

    plt.figure(figsize=(8, 6))
    plt.pie(scores, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Tipe Sampah pada Gambar')
    plt.show()

    # Display Original Image
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    plt.title('Gambar Asli')
    plt.axis('off')
    plt.show()

def select_image():
    app = QApplication(sys.argv)
    options = QFileDialog.Options()
    file_path, _ = QFileDialog.getOpenFileName(None, "Pilih Gambar", "", "Image Files (*.png;*.jpg;*.jpeg;*.bmp)", options=options)
    
    if file_path:
        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to load image from {file_path}")
            return
        detect_and_classify_trash(image)

    app.quit()

# Jalankan untuk memilih gambar
if __name__ == '__main__':
    select_image()

