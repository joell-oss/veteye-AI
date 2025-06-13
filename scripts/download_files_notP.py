# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import fitz
import os
import requests

"""
Skrypt do automatycznego pobierania i wyodrębniania obrazów USG z dokumentu PDF.
Funkcjonalność:
- Pobiera plik PDF z atlasu chorób źrebiąt ze strony edraurban.pl
- Zapisuje dokument lokalnie jako "fertility_mares.pdf"
- Tworzy strukturę folderów dla zestawu danych treningowych
- Przeszukuje wszystkie strony PDF w poszukiwaniu osadzonych obrazów
- Wyodrębnia każdy znaleziony obraz zachowując oryginalny format
- Zapisuje obrazy z nazwami "pregnant_extracted_X" lub "not_pregnant_extracted_X"
- Automatycznie numeruje pliki według kolejności znalezienia
Struktura wyjściowa:
- USG-Mares-Pregnancy-Dataset/Training/pregnant/ - obrazy ciężarnych klaczy
- USG-Mares-Pregnancy-Dataset/Training/not_pregnant/ - obrazy nieciężarnych klaczy
Wykorzystywane biblioteki:
- requests: pobieranie pliku PDF
- PyMuPDF (fitz): analiza i wyodrębnianie obrazów z PDF
- os: zarządzanie strukturą folderów
Zastosowanie: tworzenie zestawu danych do szkolenia modeli AI rozpoznających ciążę u klaczy
"""

# === Ścieżki i ustawienia ===
pdf_url = "https://edraurban.pl/layout_test/book_file/69/atlaschorobzrebiat-rozdzial1.pdf"
pdf_path = "fertility_mares.pdf"
#output_folder = "USG-Mares-Pregnancy-Dataset/Training/not_pregnant"
output_folder = "USG-Mares-Pregnancy-Dataset/Training/pregnant"
os.makedirs(output_folder, exist_ok=True)

# === Pobierz PDF ===
print("Pobieranie PDF...")
response = requests.get(pdf_url)
with open(pdf_path, "wb") as f:
    f.write(response.content)
print(f"Zapisano PDF jako: {pdf_path}")

# === Wyodrębnianie obrazów osadzonych w PDF ===
print("Wyodrębnianie obrazów z PDF...")
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
        print(f"Zapisano: {image_filename}")

doc.close()
print(f"Gotowe. Zapisano {img_count} obrazów USG.")

