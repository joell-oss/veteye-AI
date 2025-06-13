# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import os
import random
import shutil

# Ścieżki
train_folder = "USG-Mares-Pregnancy-Dataset/Training/not_pregnant"
test_folder = "USG-Mares-Pregnancy-Dataset/Test/not_pregnant"
os.makedirs(test_folder, exist_ok=True)

# Parametr – ile procent przenieść
test_ratio = 0.2  # 20%

# Lista plików JPG
files = [f for f in os.listdir(train_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(files)

# Wybór plików testowych
test_count = int(len(files) * test_ratio)
test_files = files[:test_count]

# Kopiowanie do folderu testowego
for f in test_files:
    src = os.path.join(train_folder, f)
    dst = os.path.join(test_folder, f)
    shutil.move(src, dst)  # kopiujemy, nie przenosimy

print(f"✔ Skopiowano {test_count} zdjęć do katalogu testowego.")

