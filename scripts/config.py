# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import os
import datetime
"""
Plik konfiguracyjny zawierający wszystkie stałe i parametry projektu analizy USG klaczy.
KONFIGURACJA ŚRODOWISKA:
- Wyciszenie komunikatów diagnostycznych TensorFlow
- Parametry modelu: rozmiar obrazu, wielkość partii, liczba epok
- Stopy uczenia dla treningu podstawowego i dostrajania
STRUKTURA KATALOGÓW:
- Automatyczne tworzenie hierarchii folderów projektu
- Ścieżki do danych treningowych, testowych i wyników
- Katalogi na modele, logi i raporty
PARAMETRY MODELI:
- Nazwy i lokalizacje plików modeli (bazowy, dostrojony, końcowy)
- Model szacowania dni ciąży z przedziałami czasowymi
- Parametry analizy cech obrazu (regiony, progi wykrywania)
USTAWIENIA LOGOWANIA:
- Automatyczne tworzenie plików dziennika z datą
Centralizuje wszystkie konfiguracje w jednym miejscu dla łatwego zarządzania
i modyfikacji parametrów bez ingerencji w kod główny.
"""
# Konfiguracja środowiska
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Wyciszenie mniej istotnych komunikatów TF

# Parametry modelu i treningu
IMAGE_SIZE = (380, 380)  # Rozmiar obrazu wejściowego
BATCH_SIZE = 16  # Rozmiar partii danych
EPOCHS = 60  # Liczba epok dla treningu bazowego
EPOCHS_FT = 40  # Liczba epok dla fine-tuningu
LEARNING_RATE = 1e-4  # Początkowa stopa uczenia
VALIDATION_SPLIT = 0.1  # Procent danych do walidacji

# ścieżka bazowa projektu (lokalizacja głównego katalogu projektu)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ścieżki do katalogów
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")  # Katalog z plikami Pythona
FONTS_DIR = os.path.join(BASE_DIR, "fonts")      # Katalog z czcionkami

# Pozostałe ścieżki do katalogów projektu
DATA_DIR = os.path.join(BASE_DIR, "USG-Mares-Pregnancy-Dataset")  # Główny katalog z danymi
TRAIN_DIR = os.path.join(DATA_DIR, "Training")                   # Katalog treningowy
TEST_DIR = os.path.join(DATA_DIR, "Test")                        # Katalog testowy
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")          # Katalog na punkty kontrolne
LOGS_DIR = os.path.join(BASE_DIR, "logs")                        # Katalog na logi
REPORTS_DIR = os.path.join(BASE_DIR, "wyniki")                   # Katalog na raporty
FEEDBACK_FILE = os.path.join(REPORTS_DIR, "expert_feedback.csv") # Plik feedbacku

# Utwórz katalogi, jeśli nie istnieją
for directory in [SCRIPTS_DIR, FONTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, REPORTS_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# Nazwa modelu
MODEL_NAME = f"USGEquina-Pregna_v1_0.keras"

# Utwórz katalogi, jeśli nie istnieją
for directory in [CHECKPOINTS_DIR, LOGS_DIR, REPORTS_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# Ścieżki do modeli
BASE_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, f"{MODEL_NAME}_base.keras")
FINETUNED_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, f"{MODEL_NAME}_finetuned.keras")
FINAL_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, f"{MODEL_NAME}_final.keras")
DAY_ESTIMATOR_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "pregnancy_day_estimator.keras")

# Granice dni ciąży do przewidywania
MIN_PREGNANCY_DAY = 15  # Minimalny dzień ciąży możliwy do wykrycia
MAX_PREGNANCY_DAY = 330  # Maksymalny dzień ciąży

# Parametry analizy obrazu dla szacowania dni ciąży
FEATURE_EXTRACTION_PARAMS = {
    'num_regions': 9,  # Liczba regionów do analizy
    'edge_detection_threshold': 0.1,  # Próg detekcji krawędzi
    'fluid_threshold': 0.7,  # Próg wykrywania płynu
    'tissue_threshold': 0.3   # Próg wykrywania tkanki
}

# Ustawienia dziennika
LOG_FILE = os.path.join(LOGS_DIR, f"training_log_{datetime.datetime.now().strftime('%Y%m%d')}.txt")
