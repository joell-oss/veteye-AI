# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image
from io import BytesIO

url = "https://equi-medica.pl/oferta/rozrod-koni/"
target_folder = "images"
os.makedirs(target_folder, exist_ok=True)

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

img_links = []

# Z <img src=...>
for img in soup.find_all('img'):
    src = img.get('src')
    if src and any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
        img_links.append(urljoin(url, src))

# Z <a href=...>
for a in soup.find_all('a'):
    href = a.get('href')
    if href and any(ext in href.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
        img_links.append(urljoin(url, href))

# Usuwamy duplikaty
img_links = list(set(img_links))

print(f"Znaleziono {len(img_links)} obrazów do pobrania.")

# Pobieranie i zapisywanie
for idx, img_url in enumerate(img_links):
    try:
        response = requests.get(img_url)
        img_data = response.content
        ext = os.path.splitext(img_url)[1].lower()

        # Konwersja webp → jpg
        if ext == '.webp':
            img = Image.open(BytesIO(img_data)).convert("RGB")
            filename = os.path.join(target_folder, f"mare_{idx+1}.jpg")
            img.save(filename, "JPEG", quality=95)
            absolute_path = os.path.abspath(relative_path)


        else:
            filename = os.path.join(target_folder, f"mare_{idx+1}.jpg")
            with open(filename, 'wb') as f:
                f.write(img_data)

        print(f"✔ Pobrano: {filename}")
        print(absolute_path)
    except Exception as e:
        print(f"❌ Błąd pobierania {img_url}: {e}")

