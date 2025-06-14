# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import os
import cv2
import subprocess

"""
Skrypt do pobierania filmu USG z YouTube i ekstrakcji klatek treningowych.
Funkcjonalność:
- Pobiera film z YouTube zawierający badanie USG klaczy w ciąży
- Wykorzystuje yt-dlp do ściągnięcia materiału w formacie MP4
- Automatycznie tworzy strukturę folderów dla zestawu danych
- Ekstraktuje klatki co 1 sekundę z pobranego materiału wideo
- Zapisuje każdą klatkę jako obraz JPEG z numeracją sekwencyjną
- Oblicza liczbę klatek na sekundę (FPS) dla precyzyjnego próbkowania
Parametry konfiguracyjne:
- video_url: link do filmu YouTube z badaniem USG
- download_folder: folder do zapisania pobranego filmu
- frames_folder: folder docelowy dla wyodrębnionych klatek
- interval: częstotliwość ekstrakcji (co sekundę filmu)
Struktura wyjściowa:
- usg_videos/ - pobrane filmy źródłowe
- USG-Mares-Pregnancy-Dataset/Training/not_pregnant/ - klatki treningowe
Wykorzystywane narzędzia:
- yt-dlp: pobieranie materiałów z YouTube
- OpenCV: analiza wideo i ekstrakcja klatek
- subprocess: wywołanie zewnętrznych programów
Zastosowanie: budowanie zestawu danych obrazów USG do treningu modeli AI
"""

# === KONFIGURACJA ===
video_url = "https://www.youtube.com/watch?v=JrU9F9e8lFI" # <- link USG klaczy w ciąży
download_folder = "usg_videos"
frames_folder = "USG-Mares-Pregnancy-Dataset/Training/not_pregnant"

os.makedirs(download_folder, exist_ok=True)
os.makedirs(frames_folder, exist_ok=True)

video_path = os.path.join(download_folder, "pregnant_video.mp4")

# === POBIERANIE FILMU PRZEZ yt-dlp ===
print("Pobieranie filmu z YouTube przez yt-dlp...")

try:
    subprocess.run([
        "yt-dlp",
        "-f", "mp4",
        "-o", video_path,
        video_url
    ], check=True)
except subprocess.CalledProcessError as e:
    print("Błąd pobierania filmu:", e)
    exit()

print(f"Film zapisany jako: {video_path}")

# === EKSTRAKCJA KLATEK CO 1 SEKUNDA ===
print("Rozpoczynanie ekstrakcji klatek...")

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(fps)  # co 1 sekunda

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % interval == 0:
        filename = os.path.join(frames_folder, f"pregnant_frame_{saved_count+1}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1
    frame_count += 1

cap.release()
print(f"Zapisano {saved_count} klatek do: {frames_folder}")

