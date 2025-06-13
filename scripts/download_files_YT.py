# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import os
import cv2
import subprocess

# === KONFIGURACJA ===
video_url = "https://www.youtube.com/watch?v=JrU9F9e8lFI" # <- działający link USG klaczy w ciąży
download_folder = "usg_videos"
frames_folder = "USG-Mares-Pregnancy-Dataset/Training/not_pregnant"

os.makedirs(download_folder, exist_ok=True)
os.makedirs(frames_folder, exist_ok=True)

video_path = os.path.join(download_folder, "pregnant_video.mp4")

# === POBIERANIE FILMU PRZEZ yt-dlp ===
print("🔽 Pobieranie filmu z YouTube przez yt-dlp...")

try:
    subprocess.run([
        "yt-dlp",
        "-f", "mp4",
        "-o", video_path,
        video_url
    ], check=True)
except subprocess.CalledProcessError as e:
    print("❌ Błąd pobierania filmu:", e)
    exit()

print(f"✔ Film zapisany jako: {video_path}")

# === EKSTRAKCJA KLATEK CO 1 SEKUNDA ===
print("🧪 Rozpoczynanie ekstrakcji klatek...")

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
print(f"✔ Zapisano {saved_count} klatek do: {frames_folder}")

