# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import os
import argparse
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from config import (
    TRAIN_DIR, TEST_DIR, CHECKPOINTS_DIR, LOGS_DIR, REPORTS_DIR,
    IMAGE_SIZE, BATCH_SIZE, EPOCHS, EPOCHS_FT, LEARNING_RATE, MODEL_NAME
)
from logging_utils import log_info, log_error, log_section, log_success, setup_logging
from data_loader import load_data
from model_builder import create_pregnancy_detection_model, apply_fine_tuning
from model_training import train_pregnancy_model, create_callbacks, train_with_error_handling
from evaluation import evaluate_pregnancy_model

def parse_arguments():
    """
    Parsuje i konfiguruje argumenty wiersza poleceń dla treningu modelu wykrywania ciąży.
    Definiuje wszystkie parametry sterujące procesem treningu modelu:
    - Ścieżki do katalogów z danymi treningowymi i testowymi
    - Parametry obrazów (rozmiar, rozmiar partii danych)
    - Ustawienia treningu (liczba epok, stopa uczenia)
    - Opcje wznowienia treningu z punktu kontrolnego
    Zwraca:
        argparse.Namespace: Obiekt zawierający wszystkie sparsowane argumenty
                           z wartościami domyślnymi lub podanymi przez użytkownika
    Uwagi:
        Wszystkie parametry mają wartości domyślne zdefiniowane jako stałe globalne,
        co umożliwia uruchomienie treningu bez podawania argumentów.
    """
    parser = argparse.ArgumentParser(description="Trening modelu wykrywania ciąży u klaczy")
    parser.add_argument("--train_dir", default=TRAIN_DIR, help="Katalog z danymi treningowymi")
    parser.add_argument("--test_dir", default=TEST_DIR, help="Katalog z danymi testowymi")
    parser.add_argument("--image_size", type=int, nargs=2, default=IMAGE_SIZE, help="Rozmiar obrazu (szerokość wysokość)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Rozmiar batcha")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Liczba epok dla modelu bazowego")
    parser.add_argument("--epochs_ft", type=int, default=EPOCHS_FT, help="Liczba epok dla fine-tuningu")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Początkowa stopa uczenia")
    parser.add_argument("--resume", action="store_true", help="Wznów trening od ostatniego punktu kontrolnego")
    
    return parser.parse_args()

def main():
    """
    Główna funkcja systemu wykrywania ciąży u klaczy metodą klasyfikacji binarnej.
    Przeprowadza kompletny cykl treningu modelu uczenia maszynowego:
    - Parsuje parametry konfiguracyjne i sprawdza dostępność danych
    - Wczytuje i przygotowuje zestawy danych treningowych i testowych
    - Tworzy lub wznawia model sieci neuronowej do klasyfikacji binarnej
    - Wykonuje trening bazowy oraz proces dostrajania parametrów
    - Przeprowadza szczegółową ewaluację dokładności klasyfikacji
    - Zapisuje wytrenowany model oraz szczegółowe raporty wyników
    Zwraca:
        int: Kod wyjścia programu (0 - pomyślne zakończenie, 1 - wystąpił błąd)
    Uwagi:
        Model rozróżnia dwa stany: obecność i brak ciąży na podstawie analizy
        obrazów USG przy użyciu technik głębokiego uczenia maszynowego.
    """
    # Parsuj argumenty
    args = parse_arguments()
    
    # Skonfiguruj logowanie
    log_file = setup_logging()
    
    log_section("Trening modelu wykrywania ciąży u klaczy", log_file)
    
    # Wyświetl parametry
    log_info(f"Katalog treningowy: {args.train_dir}", log_file)
    log_info(f"Katalog testowy: {args.test_dir}", log_file)
    log_info(f"Rozmiar obrazu: {args.image_size}, Batch size: {args.batch_size}", log_file)
    log_info(f"Liczba epok: {args.epochs} (bazowy) + {args.epochs_ft} (fine-tuning)", log_file)
    log_info(f"Stopa uczenia: {args.learning_rate}", log_file)
    
    # Sprawdź katalogi danych
    if not os.path.exists(args.train_dir) or not os.path.exists(args.test_dir):
        log_error(f"Katalogi danych '{args.train_dir}' lub '{args.test_dir}' nie istnieją!", log_file)
        return 1
    
    try:
        # Ładowanie danych
        log_section("Ładowanie danych treningowych", log_file)
        train_generator, test_generator, steps_per_epoch, validation_steps = load_data(
            args.train_dir,
            args.test_dir,
            args.image_size,
            args.batch_size,
            log_file
        )
        
        # Ścieżki do punktów kontrolnych
        checkpoint_base = os.path.join(CHECKPOINTS_DIR, f"{MODEL_NAME}_base.keras")
        checkpoint_ft = os.path.join(CHECKPOINTS_DIR, f"{MODEL_NAME}_finetuned.keras")
        
        # Sprawdź, czy wznowić trening
        if args.resume and os.path.exists(checkpoint_base):
            log_info(f"Wznowienie treningu z istniejącego punktu kontrolnego: {checkpoint_base}", log_file)
            from tensorflow.keras.models import load_model
            model = load_model(checkpoint_base)
        else:
            # Tworzenie modelu
            log_section("Tworzenie modelu", log_file)
            model = create_pregnancy_detection_model(
                args.image_size,
                num_classes=1,  # Klasyfikacja binarna
                learning_rate=args.learning_rate,
                log_file=log_file
            )
        
        # Podsumowanie modelu
        model.summary()
        
        # Trening modelu
        log_section("Rozpoczęcie treningu", log_file)
        history_base, history_ft, model = train_pregnancy_model(
            model,
            train_generator,
            test_generator,
            steps_per_epoch,
            validation_steps,
            checkpoint_base,
            checkpoint_ft,
            log_file
        )
        
        # Ewaluacja modelu
        log_section("Ewaluacja modelu", log_file)
        class_names = list(train_generator.class_indices.keys())
        
        # Utwórz katalog na raporty ewaluacji
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_dir = os.path.join(REPORTS_DIR, f"evaluation_{timestamp}")
        os.makedirs(evaluation_dir, exist_ok=True)
        
        # Przeprowadź ewaluację
        results = evaluate_pregnancy_model(model, test_generator, class_names, evaluation_dir, log_file)
        
        if results:
            # Zapisz model końcowy
            final_model_path = os.path.join(CHECKPOINTS_DIR, f"{MODEL_NAME}_final.keras")
            model.save(final_model_path)
            log_success(f"Model końcowy zapisany: {final_model_path}", log_file)
            
            # Zapisz wyniki jako JSON
            results_path = os.path.join(evaluation_dir, "results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
            
            log_success("Trening i ewaluacja zakończone pomyślnie", log_file)
        else:
            log_error("Ewaluacja modelu nie powiodła się", log_file=log_file)
        
        return 0
    
    except Exception as e:
        log_error(f"Błąd podczas treningu modelu: {e}", log_file=log_file)
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
