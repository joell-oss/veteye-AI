# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import fitz  # PyMuPDF
import os
import requests

# === Ścieżki i ustawienia ===
pdf_url = "https://edraurban.pl/layout_test/book_file/69/atlaschorobzrebiat-rozdzial1.pdf"
pdf_path = "fertility_mares.pdf"
#output_folder = "USG-Mares-Pregnancy-Dataset/Training/not_pregnant"
output_folder = "USG-Mares-Pregnancy-Dataset/Training/pregnant"
os.makedirs(output_folder, exist_ok=True)

# === Pobierz PDF ===
print("🔽 Pobieranie PDF...")
response = requests.get(pdf_url)
with open(pdf_path, "wb") as f:
    f.write(response.content)
print(f"✔ Zapisano PDF jako: {pdf_path}")

# === Wyodrębnianie obrazów osadzonych w PDF ===
print("🧪 Wyodrębnianie obrazów z PDF...")
doc = fitz.open(pdf_path)
img_count = 0

for page_index in range(len(doc)):
    for img_index, img in enumerate(doc.get_page_images(page_index)):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        img_count += 1

        #image_filename = os.path.join(output_folder, f"not_pregnant_extracted_{img_count}.{image_ext}")
        image_filename = os.path.join(output_folder, f"pregnant_extracted_{img_count}.{image_ext}")
        with open(image_filename, "wb") as img_file:
            img_file.write(image_bytes)
        print(f"✔ Zapisano: {image_filename}")

doc.close()
print(f"✅ Gotowe. Zapisano {img_count} obrazów USG.")

