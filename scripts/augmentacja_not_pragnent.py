# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import os
import cv2

source_folder = "USG-Mares-Pregnancy-Dataset/Training/not_pregnant"
output_folder = source_folder
os.makedirs(output_folder, exist_ok=True)

datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True,
    fill_mode='nearest'
)

for file in os.listdir(source_folder):
    if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(source_folder, file)
    img = load_img(img_path)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # ile augmentowanych kopii chcesz stworzyć z każdego zdjęcia
    N = 3

    gen = datagen.flow(x, batch_size=1)
    for i in range(N):
        aug_img = next(gen)[0].astype(np.uint8)
        out_name = f"aug1_{file[:-4]}_{i+1}.jpg"
        out_path = os.path.join(output_folder, out_name)
        cv2.imwrite(out_path, aug_img)

print("✅ Augmentacja zakończona.")

