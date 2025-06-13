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
    """Tworzy model wykrywania ciąży wykorzystujący InceptionV3 z transfer learning"""
    
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
            layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.0001)),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu', kernel_regularizer=l2(0.0001)),
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
    """Tworzy model szacowania dnia ciąży oparty na InceptionV3"""
    
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
    """Odmrażanie i dostrajanie warstw modelu bazowego"""
    
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
