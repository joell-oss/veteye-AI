# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.regularizers import l2
from logging_utils import log_info, log_error, log_section
from config import IMAGE_SIZE, LEARNING_RATE

def create_pregnancy_detection_model(input_shape=IMAGE_SIZE, num_classes=1, learning_rate=LEARNING_RATE, log_file=None):
    """
    Tworzy model sieci neuronowej do wykrywania ciąży wykorzystujący architekturę InceptionV3.
    Funkcja buduje głęboką sieć konwolucyjną opartą na pretrenowanym modelu InceptionV3
    z zastosowaniem techniki uczenia transferowego. Model bazowy pozostaje zamrożony,
    a na jego szczycie dodawane są dodatkowe warstwy klasyfikacyjne dostosowane do
    diagnostyki weterynaryjnej. Obsługuje zarówno klasyfikację binarną (ciąża/brak ciąży)
    jak i wieloklasową w zależności od potrzeb.
    Argumenty:
       input_shape: Wymiary obrazów wejściowych (wysokość, szerokość)
       num_classes: Liczba klas (1 dla binarnej, >1 dla wieloklasowej)
       learning_rate: Współczynnik uczenia dla optymalizatora
       log_file: Plik dziennika do rejestrowania procesu
    Zwraca:
       Skompilowany model Keras gotowy do treningu z odpowiednimi metrykami
       i funkcją straty dostosowaną do typu klasyfikacji
    Architektura zawiera regularyzację L2, warstwy dropout oraz normalizację
    wsadową dla zapobiegania przeuczeniu i poprawy stabilności treningu.
    """
    
    log_section("Tworzenie modelu wykrywania ciąży", log_file)
    
    try:
        # Załadowanie modelu bazowego bez warstwy klasyfikacyjnej
        base_model = applications.InceptionV3(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape + (3,)
        )
        
        # Zamrożenie wszystkich warstw modelu bazowego
        base_model.trainable = False
        
        # Określenie metryk w zależności od typu klasyfikacji
        if num_classes == 1:  # Binary classification
            metrics = [
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')
            ]
            loss = 'binary_crossentropy'
            activation = 'sigmoid'
        else:  # Multi-class classification
            metrics = [
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
            loss = 'categorical_crossentropy'
            activation = 'softmax'
        
        # Budowa modelu z modelem bazowym
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(768, activation='relu', kernel_regularizer=l2(0.0005)),  # Większa warstwa
            layers.Dropout(0.5),  # Wyższy dropout dla większego zbioru danych
            layers.Dense(384, activation='relu', kernel_regularizer=l2(0.0005)),  # Dodatkowa warstwa
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation=activation)
        ])
        
        # Kompilacja modelu
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        log_info(f"Model wykrywania ciąży utworzony", log_file)
        
        return model
    
    except Exception as e:
        log_error("Błąd podczas tworzenia modelu wykrywania ciąży", e, log_file)
        raise

def create_day_estimation_model(input_shape=IMAGE_SIZE, num_days=316, learning_rate=LEARNING_RATE/2, log_file=None):
    """
    Model głębokiego uczenia do klasyfikacji obrazów medycznych z wykorzystaniem transfer learning.
    Wykorzystuje przedtrénowany model InceptionV3 jako ekstraktor cech, rozszerzony o warstwy
    gęste z regularyzacją L2 i dropout. Model klasyfikuje obraz do jednej z 316 możliwych
    kategorii używając aktywacji softmax.
    Architektura:
    - InceptionV3 (zamrożony) - ekstrakcja cech z ImageNet
    - GlobalAveragePooling2D - redukcja wymiarowości  
    - 3 warstwy Dense (1536→768→256) z regularyzacją L2 i dropout
    - Warstwa wyjściowa softmax dla 316 klas
    Parametry:
    - learning_rate: zmniejszona o połowę względem bazowej wartości
    - loss: sparse_categorical_crossentropy dla etykiet całkowitych
    - metryki: accuracy i top-5 accuracy dla oceny wydajności
    """
    
    log_section("Tworzenie modelu szacowania dnia ciąży", log_file)
    
    try:
        # Załadowanie modelu bazowego bez warstwy klasyfikacyjnej
        base_model = applications.InceptionV3(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape + (3,)
        )
        
        # Zamrożenie wszystkich warstw modelu bazowego
        base_model.trainable = False
        
        # Budowa modelu z modelem bazowym - głęboka sieć
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(1536, activation='relu', kernel_regularizer=l2(0.0001)),
            layers.Dropout(0.5),
            layers.Dense(768, activation='relu', kernel_regularizer=l2(0.0001)),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu', kernel_regularizer=l2(0.0001)),
            layers.Dropout(0.3),
            layers.Dense(num_days, activation='softmax')  # Klasyfikacja wieloklasowa dni ciąży
        ])
        
        # Kompilacja modelu
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',  # Dla etykiet całkowitych
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
        )
        
        log_info(f"Model szacowania dnia ciąży utworzony z {num_days} możliwymi dniami", log_file)
        
        return model
    
    except Exception as e:
        log_error("Błąd podczas tworzenia modelu szacowania dnia ciąży", e, log_file)
        raise

def apply_fine_tuning(model, learning_rate=LEARNING_RATE/10, log_file=None):
    """
    Funkcja do dostrajania pretrenowanego modelu przez odmrożenie części warstw.
    Implementuje technikę fine-tuning poprzez:
    1. Odmrożenie ostatnich 30% warstw modelu bazowego (InceptionV3)
    2. Rekompilację modelu z obniżoną stopą uczenia (1/10 bazowej wartości)
    3. Automatyczne dostosowanie funkcji straty i metryk według typu modelu
    Typy modeli:
    - Binarny (1 neuron wyjściowy): wykrywanie obecności cechy
    - Wieloklasowy (>2 neurony): klasyfikacja lub szacowanie wartości
    - Specjalny przypadek: modele z wieloma klasami używają top-5 accuracy
    Parametry dostrajania:
    - Stopień odmrożenia: 30% najgłębszych warstw
    - Stopa uczenia: zmniejszona 10-krotnie dla stabilności
    - Funkcja straty: automatycznie dobrana do architektury wyjściowej
    Zapewnia stopniowe dostrajanie cech wysokopoziomowych przy zachowaniu
    wcześniej wyuczonych reprezentacji niskopoziomowych.
    """ 
    
    log_section("Rozpoczynanie fine-tuningu modelu", log_file)
    
    try:
        # Odmrożenie ostatnich bloków modelu bazowego
        base_model = model.layers[0]
        
        # Odmrażamy ostatnie 30% warstw
        num_layers = len(base_model.layers)
        unfreeze_layers = int(num_layers * 0.3)
        
        # Zamrażamy najpierw wszystkie warstwy, potem odmrażamy ostatnie
        for layer in base_model.layers:
            layer.trainable = False
            
        for layer in base_model.layers[-unfreeze_layers:]:
            layer.trainable = True
        
        log_info(f"Odmrożono {unfreeze_layers} z {num_layers} warstw modelu bazowego", log_file)
        
        # Sprawdzenie typu modelu (wykrywania ciąży czy szacowania dnia)
        is_binary = model.layers[-1].units == 1
        is_day_estimator = model.layers[-1].units > 2
        
        if is_binary:  # Model wykrywania ciąży (binarny)
            metrics = [
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')
            ]
            loss = 'binary_crossentropy'
        elif is_day_estimator:  # Model szacowania dnia ciąży (wieloklasowy)
            metrics = [
                'accuracy',
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')
            ]
            loss = 'sparse_categorical_crossentropy'
        else:  # Inny model wieloklasowy
            metrics = [
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
            loss = 'categorical_crossentropy'
        
        # Rekompilacja modelu z niższą stopą uczenia
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        log_info(f"Model przekompilowany do fine-tuningu z LR={learning_rate}", log_file)
        
        return model
    
    except Exception as e:
        log_error("Błąd podczas fine-tuningu modelu", e, log_file)
        raise
