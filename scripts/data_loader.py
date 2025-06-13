# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import inception_v3
from logging_utils import log_info, log_error, log_section
from config import IMAGE_SIZE, BATCH_SIZE

def load_data(train_dir, test_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, log_file=None):
    """
    Ładuje i przygotowuje dane obrazowe do treningu modelu wykrywania ciąży u klaczy.
    Funkcjonalność:
    - Tworzy generatory danych z preprocessing dla modelu Inception V3
    - Stosuje augmentację danych treningowych (obroty, przesunięcia, odbicia, jasność)
    - Przygotowuje dane testowe bez augmentacji dla obiektywnej oceny
    - Konfiguruje klasyfikację binarną (ciąża/brak ciąży)
    - Oblicza liczbę kroków na epokę i walidację
    Parametry:
    - train_dir: katalog z obrazami treningowymi
    - test_dir: katalog z obrazami testowymi  
    - image_size: docelowy rozmiar obrazów
    - batch_size: wielkość partii danych
    - log_file: plik do zapisu logów
    Zwraca:
    Generatory danych oraz parametry treningu (kroki na epokę, kroki walidacji).
    Automatycznie rozpoznaje strukturę katalogów i klasy obrazów.
    """
    
    log_section("Ładowanie danych", log_file)
    
    try:
        # Generator danych z augmentacją dla zbioru treningowego
        train_datagen = ImageDataGenerator(
            preprocessing_function=inception_v3.preprocess_input,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        
        # Generator danych bez augmentacji dla zbioru testowego
        test_datagen = ImageDataGenerator(
            preprocessing_function=inception_v3.preprocess_input
        )
        
        # Ładowanie danych treningowych
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='binary',  # Klasyfikacja binarna: ciąża / brak ciąży
            shuffle=True,
            seed=42
        )
        
        # Ładowanie danych testowych
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        log_info(f"Znaleziono {train_generator.num_classes} klasy: {train_generator.class_indices}", log_file)
        log_info(f"Dane treningowe: {train_generator.samples} obrazów", log_file)
        log_info(f"Dane testowe: {test_generator.samples} obrazów", log_file)
        
        # Obliczenie kroków na epokę i walidację
        steps_per_epoch = train_generator.samples // batch_size
        validation_steps = test_generator.samples // batch_size
        
        log_info(f"Kroki na epokę: {steps_per_epoch}", log_file)
        log_info(f"Kroki walidacji: {validation_steps}", log_file)
        
        return train_generator, test_generator, steps_per_epoch, validation_steps
    
    except Exception as e:
        log_error("Błąd podczas ładowania danych", e, log_file)
        raise

def create_day_estimation_dataset(data_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, log_file=None):
    """
    Tworzy zestaw danych do treningu modelu szacującego dzień ciąży u klaczy.
    Funkcjonalność:
    - Ładuje obrazy z katalogów nazwanych według dni ciąży (np. "30", "45", "120")
    - Przygotowuje dane z łagodną augmentacją dostosowaną do analizy rozwoju płodu
    - Automatycznie dzieli dane na zbiór treningowy (80%) i walidacyjny (20%)
    - Tworzy mapowanie między indeksami klas a rzeczywistymi dniami ciąży
    - Konfiguruje klasyfikację wieloklasową (jeden dzień = jedna klasa)
    Parametry:
    - data_dir: katalog główny zawierający podkatalogi z dniami ciąży
    - image_size: docelowy rozmiar obrazów
    - batch_size: wielkość partii danych
    - log_file: plik do zapisu logów
    Zwraca:
    Generatory danych treningowych i walidacyjnych oraz mapowanie dni ciąży.
    Umożliwia trenowanie modelu do precyzyjnego określania zaawansowania ciąży
    - nie uwzględnione w demonstratorze.
    """
    try:
        log_section("Tworzenie zestawu danych do szacowania dni ciąży", log_file)
        
        # Sprawdzenie, czy katalog istnieje
        if not os.path.exists(data_dir):
            log_error(f"Katalog danych dni ciąży {data_dir} nie istnieje", log_file=log_file)
            return None, None
        
        # Generator danych z augmentacją dla zbioru treningowego
        datagen = ImageDataGenerator(
            preprocessing_function=inception_v3.preprocess_input,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            brightness_range=[0.9, 1.1],
            horizontal_flip=True,
            validation_split=0.2  # Część danych do walidacji
        )
        
        # Ładowanie danych treningowych
        train_generator = datagen.flow_from_directory(
            data_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='sparse',  # Etykiety jako liczby
            shuffle=True,
            seed=42,
            subset='training'  # Podzbiór treningowy
        )
        
        # Ładowanie danych walidacyjnych
        validation_generator = datagen.flow_from_directory(
            data_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=False,
            subset='validation'  # Podzbiór walidacyjny
        )
        
        # Mapowanie indeksów klas na dni ciąży
        class_indices = train_generator.class_indices
        day_mapping = {v: int(k) for k, v in class_indices.items() if k.isdigit()}
        
        log_info(f"Utworzono zestaw danych do szacowania dni ciąży", log_file)
        log_info(f"Liczba klas (dni ciąży): {len(class_indices)}", log_file)
        log_info(f"Liczba obrazów treningowych: {train_generator.samples}", log_file)
        log_info(f"Liczba obrazów walidacyjnych: {validation_generator.samples}", log_file)
        
        return train_generator, validation_generator, day_mapping
    
    except Exception as e:
        log_error("Błąd podczas tworzenia zestawu danych do szacowania dni ciąży", e, log_file)
        raise

def load_and_preprocess_image(image_path, image_size=IMAGE_SIZE):
    """
    Ładuje i przetwarza pojedynczy obraz USG do analizy przez model uczenia maszynowego.
    Proces przetwarzania:
    - Wczytuje obraz z pliku i zmienia jego rozmiar do wymaganych wymiarów
    - Konwertuje obraz na tablicę numeryczną z wartościami pikseli
    - Stosuje preprocessing specyficzny dla modelu Inception V3 (normalizacja)
    - Dodaje wymiar partii (batch dimension) wymagany przez model
    Parametry:
    - image_path: ścieżka do pliku obrazu
    - image_size: docelowy rozmiar obrazu (szerokość, wysokość)
    Zwraca:
    - img_expanded: obraz gotowy do predykcji (z wymiarem partii)
    - img_array: obraz przetworzony bez wymiaru partii (do wizualizacji)
    Obsługuje błędy ładowania i zwraca None w przypadku problemów.
    Zapewnia kompatybilność z wymaganiami modelu głębokiego uczenia.
    """
    try:
        # Załadowanie obrazu
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        # Przetworzenie obrazu z użyciem preprocessingu InceptionV3
        img_array = inception_v3.preprocess_input(img_array)
        
        # Dostosowanie wymiarów dla modelu (batch_size=1)
        img_expanded = np.expand_dims(img_array, axis=0)
        
        return img_expanded, img_array
    
    except Exception as e:
        log_error(f"Błąd podczas ładowania obrazu {image_path}", e)
        return None, None
