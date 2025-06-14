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
    TRAIN_DIR, IMAGE_SIZE, BATCH_SIZE, EPOCHS, EPOCHS_FT, LEARNING_RATE,
    CHECKPOINTS_DIR, LOGS_DIR, REPORTS_DIR, DAY_ESTIMATOR_MODEL_PATH
)
from logging_utils import log_info, log_error, log_section, log_success, setup_logging
from data_loader import create_day_estimation_dataset
from model_builder import create_day_estimation_model, apply_fine_tuning
from model_training import train_day_estimation_model
from evaluation import evaluate_day_estimation_model

def parse_arguments():
    """
    Analizuje i przetwarza argumenty wiersza poleceń dla procesu treningu modelu szacowania dnia ciąży.
    Funkcja konfiguruje parser argumentów umożliwiający dostosowanie parametrów treningu 
    sieci neuronowej. Obejmuje ustawienia katalogu danych, wymiarów obrazów, rozmiaru 
    partii danych, liczby epok dla treningu podstawowego i dostrajania, stopy uczenia 
    oraz opcję wznawiania treningu od zapisanego punktu kontrolnego.
    Zwraca:
       argparse.Namespace: Obiekt zawierający sparsowane parametry konfiguracyjne treningu
    Uwagi:
       Wymaga obowiązkowego wskazania katalogu z danymi treningowymi zawierającego 
       podkatalogi numerowane według dni ciąży. Pozostałe parametry mają wartości 
       domyślne zdefiniowane w stałych konfiguracyjnych programu.
    """
    parser = argparse.ArgumentParser(description="Trening modelu szacowania dnia ciąży u klaczy")
    parser.add_argument("--data_dir", required=True, help="Katalog z podkatalogami dni ciąży")
    parser.add_argument("--image_size", type=int, nargs=2, default=IMAGE_SIZE, help="Rozmiar obrazu (szerokość wysokość)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Rozmiar batcha")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Liczba epok dla modelu bazowego")
    parser.add_argument("--epochs_ft", type=int, default=EPOCHS_FT, help="Liczba epok dla fine-tuningu")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE/2, help="Początkowa stopa uczenia")
    parser.add_argument("--resume", action="store_true", help="Wznów trening od ostatniego punktu kontrolnego")
    
    return parser.parse_args()

def main():
    """
    Główna funkcja systemu szacowania dnia ciąży u klaczy na podstawie analizy obrazów USG.
    Funkcja przeprowadza kompletny proces treningu modelu uczenia maszynowego:
    - Wczytuje i przetwarza dane treningowe z obrazów USG
    - Tworzy lub wznawia model sieci neuronowej do klasyfikacji dni ciąży
    - Wykonuje trening bazowy oraz dostrajanie parametrów modelu
    - Przeprowadza ewaluację dokładności przewidywań
    - Zapisuje wytrenowany model oraz raporty z wynikami
    Zwraca:
        int: Kod wyjścia (0 - sukces, 1 - błąd)
    Uwagi:
        Model wykorzystuje techniki głębokiego uczenia do automatycznego
        rozpoznawania stadium rozwoju płodu na podstawie cech obrazowych.
    Nie realizowane w demonstratorze.    
    """
    # Parsuj argumenty
    args = parse_arguments()
    
    # Skonfiguruj logowanie
    log_file = setup_logging()
    
    log_section("Trening modelu szacowania dnia ciąży u klaczy", log_file)
    
    # Wyświetl parametry
    log_info(f"Katalog danych: {args.data_dir}", log_file)
    log_info(f"Rozmiar obrazu: {args.image_size}, Batch size: {args.batch_size}", log_file)
    log_info(f"Liczba epok: {args.epochs} (bazowy) + {args.epochs_ft} (fine-tuning)", log_file)
    log_info(f"Stopa uczenia: {args.learning_rate}", log_file)
    
    # Sprawdź katalog danych
    if not os.path.exists(args.data_dir):
        log_error(f"Katalog danych '{args.data_dir}' nie istnieje!", log_file)
        return 1
    
    try:
        # Ładowanie danych
        log_section("Ładowanie danych treningowych", log_file)
        train_generator, val_generator, day_mapping = create_day_estimation_dataset(
            args.data_dir,
            args.image_size,
            args.batch_size,
            log_file
        )
        
        if not train_generator or not day_mapping:
            log_error("Nie udało się utworzyć zestawu danych", log_file=log_file)
            return 1
        
        # Liczba dni (klas)
        num_days = len(day_mapping)
        log_info(f"Liczba różnych dni ciąży: {num_days}", log_file)
        
        # Sprawdź, czy wznowić trening
        checkpoint_path = DAY_ESTIMATOR_MODEL_PATH
        
        if args.resume and os.path.exists(checkpoint_path):
            log_info(f"Wznowienie treningu z istniejącego punktu kontrolnego: {checkpoint_path}", log_file)
            from tensorflow.keras.models import load_model
            model = load_model(checkpoint_path)
        else:
            # Tworzenie modelu
            log_section("Tworzenie modelu", log_file)
            model = create_day_estimation_model(
                args.image_size,
                num_days=num_days,
                learning_rate=args.learning_rate,
                log_file=log_file
            )
        
        # Podsumowanie modelu
        model.summary()
        
        # Trening modelu
        log_section("Rozpoczęcie treningu", log_file)
        history_base, history_ft, model = train_day_estimation_model(
            model,
            train_generator,
            val_generator,
            day_mapping,
            checkpoint_path,
            log_file
        )
        
        # Ewaluacja modelu
        log_section("Ewaluacja modelu", log_file)
        
# Utwórz katalog na raporty ewaluacji
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_dir = os.path.join(REPORTS_DIR, f"day_estimation_evaluation_{timestamp}")
        os.makedirs(evaluation_dir, exist_ok=True)
        
        # Przeprowadź ewaluację
        results = evaluate_day_estimation_model(model, val_generator, day_mapping, evaluation_dir, log_file)
        
        if results:
            # Zapisz model końcowy
            final_model_path = os.path.join(CHECKPOINTS_DIR, "pregnancy_day_estimator_final.keras")
            model.save(final_model_path)
            log_success(f"Model końcowy zapisany: {final_model_path}", log_file)
            
            # Zapisz wyniki jako JSON
            results_path = os.path.join(evaluation_dir, "day_estimation_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
            
            # Zapisz mapowanie dni ciąży
            day_mapping_path = os.path.join(CHECKPOINTS_DIR, "day_mapping.json")
            with open(day_mapping_path, 'w', encoding='utf-8') as f:
                json.dump(day_mapping, f, indent=4)
            
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
