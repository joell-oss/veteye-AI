# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import os
import time
import json
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from logging_utils import log_info, log_error, log_section, log_success, log_warning
from config import CHECKPOINTS_DIR, LOGS_DIR, EPOCHS, EPOCHS_FT

def create_callbacks(model_path, monitor='val_loss', mode='min', 
                    patience_es=10, patience_lr=5, min_lr=1e-7,
                    tensorboard_dir=None, log_file=None):
    """
    Zestaw callbacków do monitorowania i kontroli procesu treningu modelu.
    Implementuje standardowe mechanizmy optymalizacji treningu:
    - EarlyStopping: przerywa trening przy braku poprawy przez określoną liczbę epok
    - ReduceLROnPlateau: zmniejsza stopę uczenia o 70% przy stagnacji metryki
    - ModelCheckpoint: automatyczny zapis najlepszego i ostatniego stanu modelu
    - TensorBoard: opcjonalne logowanie metryk do wizualizacji
    Parametry kontroli:
    - monitor: metryka do śledzenia (domyślnie val_loss)
    - patience_es/lr: tolerancja braku poprawy (10/5 epok)
    - min_lr: minimalna stopa uczenia (1e-7)
    - mode: kierunek optymalizacji ('min' dla loss, 'max' dla accuracy)
    Zapewnia stabilny trening z automatyczną regulacją parametrów i zachowaniem
    najlepszych wyników bez konieczności ręcznej interwencji.
    """
    
    callbacks = []
    
    # Early Stopping - zatrzymuje trening gdy nie ma poprawy
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience_es,
        restore_best_weights=True,
        verbose=1,
        mode=mode
    )
    callbacks.append(early_stopping)
    
    # Redukcja LR - zmniejsza stopę uczenia gdy nie ma poprawy
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor,
        factor=0.3,
        patience=patience_lr,
        min_lr=min_lr,
        verbose=1,
        mode=mode
    )
    callbacks.append(reduce_lr)
    
    # Zapis najlepszego modelu
    model_checkpoint = ModelCheckpoint(
        model_path,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
        mode=mode
    )
    callbacks.append(model_checkpoint)
    
    # Zapis ostatniego modelu (dodatkowy checkpoint)
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path).replace('.keras', '_last.keras')
    last_checkpoint = ModelCheckpoint(
        os.path.join(model_dir, model_name),
        save_best_only=False,
        save_weights_only=False,
        verbose=0
    )
    callbacks.append(last_checkpoint)
    
    # Jeśli podano katalog TensorBoard, dodaj callback
    if tensorboard_dir:
        log_dir = os.path.join(tensorboard_dir, time.strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
        log_info(f"Logi TensorBoard będą zapisywane w {log_dir}", log_file)
    
    return callbacks

def train_with_error_handling(model, train_generator, val_generator, epochs, callbacks, 
                             steps_per_epoch=None, validation_steps=None, stage_name="trening",
                             max_retries=3, log_file=None):
    """
    Funkcja treningu modelu z mechanizmem odporności na błędy i automatycznymi powtórzeniami.
    Implementuje proces treningu przez:
    - Automatyczne obliczanie kroków na epokę jeśli nie podano jawnie
    - Obsługę błędów z możliwością ponowienia treningu (domyślnie 3 próby)
    - Wyłączenie wielowątkowości dla zwiększenia stabilności
    - Automatyczny zapis historii treningu do pliku JSON z znacznikiem czasu
    Parametry bezpieczeństwa:
    - max_retries: maksymalna liczba prób (domyślnie 3)
    - workers=0: jednowątkowe przetwarzanie danych
    - max_queue_size=10: ograniczenie kolejki generatora
    - 5-sekundowa pauza między próbami
    Funkcja zwraca obiekt historii treningu lub None w przypadku niepowodzenia
    wszystkich prób. Wszystkie operacje są logowane z odpowiednimi poziomami
    (info, success, error, warning).
    """
    
    log_section(f"Rozpoczęcie {stage_name} modelu", log_file)
    
    current_try = 0
    history = None
    
    while current_try < max_retries:
        try:
            log_info(f"Rozpoczynanie {stage_name} (próba {current_try+1}/{max_retries})...", log_file)
            
            # Jeśli nie podano kroków, oblicz je
            if steps_per_epoch is None:
                steps_per_epoch = train_generator.samples // train_generator.batch_size
            
            if validation_steps is None and val_generator:
                validation_steps = val_generator.samples // val_generator.batch_size
            
            # Trening modelu
            history = model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                callbacks=callbacks,
                verbose=1,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                workers=0,  # Wyłączenie wielowątkowości dla stabilności
                use_multiprocessing=False,
                max_queue_size=10
            )
            
            # Jeśli trening się powiódł, zapisz historię i przerwij pętlę
            if history:
                log_success(f"{stage_name.capitalize()} zakończony pomyślnie", log_file)
                
                # Zapisz historię treningu
                history_file = os.path.join(LOGS_DIR, f"history_{stage_name}_{time.strftime('%Y%m%d_%H%M')}.json")
                save_training_history(history, history_file)
                break
        
        except Exception as e:
            current_try += 1
            log_error(f"Błąd podczas {stage_name}", e, log_file)
            
            if current_try < max_retries:
                log_warning(f"Ponawianie próby treningu ({current_try}/{max_retries})...", log_file)
                time.sleep(5)  # Odczekaj 5 sekund przed ponowieniem
            else:
                log_error(f"Wszystkie próby {stage_name} nie powiodły się. Sprawdź logi błędów.", log_file=log_file)
    
    return history

def save_training_history(history, filename):
    """
    Zapisuje historię treningu modelu do pliku JSON z obsługą typów NumPy.
    Konwertuje metryki treningu z obiektów NumPy na standardowe typy Python
    dla zapewnienia kompatybilności z formatem JSON. Wszystkie wartości liczbowe
    są przekształcane na typ float, a struktura danych zachowuje oryginalne
    nazwy metryk jako klucze słownika.
    Funkcjonalność:
    - Automatyczna konwersja typów NumPy → Python float
    - Kodowanie UTF-8 dla obsługi polskich znaków
    - Formatowanie JSON z wcięciami dla czytelności
    - Podstawowa obsługa błędów z komunikatami w konsoli
    Umożliwia późniejszą analizę przebiegu treningu i porównywanie wyników
    między różnymi eksperymentami.
    """
    
    try:
        # Konwersja wartości NumPy na Python float
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(val) for val in values]
        
        # Zapisz do pliku JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history_dict, f, indent=4)
        
        print(f"Zapisano historię treningu do {filename}")
        
    except Exception as e:
        print(f"Błąd podczas zapisywania historii treningu: {e}")

def train_pregnancy_model(model, train_generator, val_generator, steps_per_epoch, validation_steps,
                         checkpoint_base, checkpoint_ft, log_file=None):
    """
    Kompletny dwuetapowy proces treningu modelu wykrywania ciąży z transfer learning.
    Implementuje strategię treningu składającą się z:
    1. Trening bazowy: zamrożone warstwy pretrenowanego modelu, uczenie tylko głowy klasyfikacyjnej
    2. Fine-tuning: odmrożenie części warstw i dostrajanie z niższą stopą uczenia
    Parametry treningu:
    - Etap bazowy: patience 12/6, standardowe LR, monitor val_loss
    - Fine-tuning: patience 15/7, LR=1e-5, min_lr=1e-8, zwiększona tolerancja
    Zarządzanie modelami:
    - Automatyczne ładowanie najlepszego modelu bazowego do fine-tuningu
    - Niezależne checkpointy dla każdego etapu treningu
    - Logowanie TensorBoard dla obu faz
    Zwraca historię obu etapów treningu oraz finalny model, umożliwiając
    analizę postępów uczenia i porównanie skuteczności strategii transfer learning.
    """
    
    # Trening modelu bazowego
    callbacks_base = create_callbacks(
        checkpoint_base,
        monitor='val_loss',
        mode='min',
        patience_es=12,
        patience_lr=6,
        tensorboard_dir=LOGS_DIR,
        log_file=log_file
    )
    
    history_base = train_with_error_handling(
        model,
        train_generator,
        val_generator,
        EPOCHS,
        callbacks_base,
        steps_per_epoch,
        validation_steps,
        "trening modelu bazowego",
        log_file=log_file
    )
    
    # Fine-tuning - załaduj najlepszy model bazowy
    if os.path.exists(checkpoint_base):
        log_info("Ładowanie najlepszego modelu bazowego do fine-tuningu", log_file)
        model = tf.keras.models.load_model(checkpoint_base)
    else:
        log_warning("Nie znaleziono najlepszego modelu bazowego - używam aktualnego", log_file)
    
    # fine-tuning
    from model_builder import apply_fine_tuning
    model = apply_fine_tuning(model, learning_rate=1e-5, log_file=log_file)
    
    # Trening z fine-tuningiem
    callbacks_ft = create_callbacks(
        checkpoint_ft,
        monitor='val_loss',
        mode='min',
        patience_es=15,
        patience_lr=7,
        min_lr=1e-8,
        tensorboard_dir=LOGS_DIR,
        log_file=log_file
    )
    
    history_ft = train_with_error_handling(
        model,
        train_generator,
        val_generator,
        EPOCHS_FT,
        callbacks_ft,
        steps_per_epoch,
        validation_steps,
        "fine-tuning modelu",
        log_file=log_file
    )
    
    return history_base, history_ft, model

def train_day_estimation_model(model, train_generator, val_generator, day_mapping,
                             checkpoint_path, log_file=None):
    """
    Kompletny proces treningu modelu szacowania dnia ciąży z dwuetapową strategią uczenia.
    Realizuje zaawansowany trening wieloklasowy składający się z:
    1. Trening bazowy: uczenie z zamrożonymi warstwami pretrenowanymi
    2. Fine-tuning: dostrajanie z odmrożonymi warstwami i obniżoną stopą uczenia (5e-6)
    Parametry optymalizacji:
    - Etap bazowy: patience 15/8, monitoring val_loss
    - Fine-tuning: patience 20/10, min_lr=1e-8, zwiększona tolerancja dla złożoności zadania
    Zarządzanie artefaktami:
    - Osobne checkpointy dla każdego etapu (_finetuned.keras)
    - Automatyczny zapis mapowania klas dni ciąży do pliku JSON
    - Logowanie TensorBoard dla wizualizacji postępów
    Specyfika zadania wieloklasowego:
    - Obsługa dużej liczby klas (dni ciąży)
    - Zachowanie mapowania etykiet dla interpretacji wyników
    - Dostosowane parametry uczenia dla stabilności treningu złożonych klasyfikatorów
    """
    
    # Trening modelu bazowego
    callbacks = create_callbacks(
        checkpoint_path,
        monitor='val_loss',
        mode='min',
        patience_es=15,
        patience_lr=8,
        tensorboard_dir=LOGS_DIR,
        log_file=log_file
    )
    
    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = val_generator.samples // val_generator.batch_size
    
    history_base = train_with_error_handling(
        model,
        train_generator,
        val_generator,
        EPOCHS,
        callbacks,
        steps_per_epoch,
        validation_steps,
        "trening modelu estymacji dni ciąży",
        log_file=log_file
    )
    
    # Fine-tuning
    if os.path.exists(checkpoint_path):
        log_info("Ładowanie najlepszego modelu do fine-tuningu", log_file)
        model = tf.keras.models.load_model(checkpoint_path)
    else:
        log_warning("Nie znaleziono najlepszego modelu - używam aktualnego", log_file)
    
    # Zastosuj fine-tuning
    from model_builder import apply_fine_tuning
    model = apply_fine_tuning(model, learning_rate=5e-6, log_file=log_file)
    
    # Trening z fine-tuningiem
    ft_checkpoint_path = checkpoint_path.replace('.keras', '_finetuned.keras')
    callbacks_ft = create_callbacks(
        ft_checkpoint_path,
        monitor='val_loss',
        mode='min',
        patience_es=20,
        patience_lr=10,
        min_lr=1e-8,
        tensorboard_dir=LOGS_DIR,
        log_file=log_file
    )
    
    history_ft = train_with_error_handling(
        model,
        train_generator,
        val_generator,
        EPOCHS_FT,
        callbacks_ft,
        steps_per_epoch,
        validation_steps,
        "fine-tuning modelu estymacji dni",
        log_file=log_file
    )
    
    # Zapis mapowanie dni ciąży
    day_mapping_file = os.path.join(CHECKPOINTS_DIR, "day_mapping.json")
    with open(day_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(day_mapping, f, indent=4)
    
    log_info(f"Zapisano mapowanie dni ciąży do {day_mapping_file}", log_file)
    
    return history_base, history_ft, model
