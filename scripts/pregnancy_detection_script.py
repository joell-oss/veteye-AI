# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, LambdaCallback
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import datetime
import time
import traceback
import re
from scipy import ndimage
import tkinter as tk
from tkinter import filedialog, Label, Entry, Button, Text, Scrollbar, Frame
from PIL import Image, ImageTk
import matplotlib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Ustawienia środowiska TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Wyciszenie mniej istotnych komunikatów TF
tf.keras.backend.clear_session()  # Wyczyszczenie poprzednich sesji

# Konfiguracja GPU, jeśli jest dostępny
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Trening będzie używał {len(gpus)} urządzeń GPU")
        # Włączenie mixed precision dla szybszego treningu na GPU
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    except RuntimeError as e:
        print(f"Błąd konfiguracji GPU: {e}")
else:
    print("Nie wykryto GPU, trening będzie używał CPU")

# Parametry globalne - dostosowane do większego zbioru danych
IMAGE_SIZE = (380, 380)  # Zwiększony rozmiar obrazu dla lepszej jakości analizy
BATCH_SIZE = 16  # Dostosowany do wielkości zbioru danych
EPOCHS = 60  # Zwiększona liczba epok dla treningu bazowego
EPOCHS_FT = 40  # Zwiększona liczba epok dla fine-tuningu
LEARNING_RATE = 8e-5  # Dostosowana stopa uczenia
VALIDATION_SPLIT = 0.1  # Mniejszy split walidacyjny ze względu na większy zbiór danych

# Ścieżki do danych
DATA_DIR = "USG-Mares-Pregnancy-Dataset"  # Główny katalog z danymi
TRAIN_DIR = os.path.join(DATA_DIR, "Training")  # Katalog treningowy
TEST_DIR = os.path.join(DATA_DIR, "Test")  # Katalog testowy
CHECKPOINTS_DIR = "checkpoints"  # Katalog na punkty kontrolne
LOGS_DIR = "logs"  # Katalog na logi TensorBoard
REPORTS_DIR = "wyniki"  # Katalog na raporty diagnostyczne
MODEL_NAME = f"usg_pregnancy_model_inceptionv3_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"  # Nazwa modelu z rozmiarem

# Utwórz katalogi, jeśli nie istnieją
for directory in [CHECKPOINTS_DIR, LOGS_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Funkcja pomocnicza do zapisu informacji o przebiegu treningu
def log_training_progress(info, log_file="training_log.txt"):
    """Zapisuje informacje o przebiegu treningu do pliku z datą i godziną"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {info}"
    print(log_message)
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")

# Funkcja do ładowania i przetwarzania danych
def load_data(train_dir, test_dir, image_size, batch_size):
    """Ładuje i przygotowuje dane treningowe i testowe z podanych katalogów"""
    
    # Generator danych z augmentacją dla zbioru treningowego
    train_datagen = ImageDataGenerator(
        preprocessing_function=applications.inception_v3.preprocess_input,
        rotation_range=20,  # Zwiększone rotacje
        width_shift_range=0.15,  # Większe przesunięcia
        height_shift_range=0.15,
        zoom_range=0.2,  # Większy zoom
        brightness_range=[0.8, 1.2],  # Większa zmiana jasności
        horizontal_flip=True,
        vertical_flip=True,  # USG często można odwracać
        fill_mode='nearest'
    )
    
    # Generator danych bez augmentacji dla zbioru testowego
    test_datagen = ImageDataGenerator(
        preprocessing_function=applications.inception_v3.preprocess_input
    )
    
    # Ładowanie danych treningowych
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',  # Klasyfikacja binarna: ciąża / brak ciąży
        shuffle=True,
        seed=42  # Stały seed dla powtarzalności
    )
    
    # Ładowanie danych testowych
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',  # Klasyfikacja binarna
        shuffle=False
    )
    
    log_training_progress(f"Znaleziono {train_generator.num_classes} klasy: {train_generator.class_indices}")
    log_training_progress(f"Dane treningowe: {train_generator.samples} obrazów")
    log_training_progress(f"Dane testowe: {test_generator.samples} obrazów")
    
    # Obliczenie kroków na epokę i walidację
    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = test_generator.samples // batch_size
    
    log_training_progress(f"Kroki na epokę: {steps_per_epoch}")
    log_training_progress(f"Kroki walidacji: {validation_steps}")
    
    return train_generator, test_generator, steps_per_epoch, validation_steps

# Funkcja tworząca model z InceptionV3 jako bazą
def create_transfer_model(input_shape, num_classes=1):
    """Tworzy model wykorzystujący InceptionV3 z transfer learning"""
    
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
    
    # Budowa modelu z modelem bazowym - dodatkowe warstwy dla lepszego uczenia
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss,
        metrics=metrics
    )
    
    return model

# Funkcja do fine-tuningu modelu
def apply_fine_tuning(model, learning_rate=0.00001):
    """Odmrażanie i dostrajanie ostatnich warstw modelu bazowego"""
    
    # Odmrożenie ostatnich bloków modelu InceptionV3
    base_model = model.layers[0]
    
    # Możemy odmrozić ostatnie 3 bloki (ok. 30% warstw)
    for layer in base_model.layers[:-60]:  # Odmrażamy więcej warstw
        layer.trainable = False
    for layer in base_model.layers[-60:]:
        layer.trainable = True
    
    log_training_progress(f"Odmrożono {sum(1 for layer in base_model.layers if layer.trainable)} z {len(base_model.layers)} warstw modelu bazowego")
    
    # Określenie typu klasyfikacji i odpowiednich metryk
    is_binary = model.layers[-1].units == 1
    
    if is_binary:  # Binary classification
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')
        ]
        loss = 'binary_crossentropy'
    else:  # Multi-class classification
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
    
    return model

# Funkcja do wizualizacji historii treningu z obsługą różnych nazw metryk
def plot_training_history(history, title, filename):
    """Tworzy i zapisuje wykresy z historii treningu z obsługą różnych nazw metryk"""
    
    # Sprawdzenie dostępnych metryk w historii
    available_metrics = list(history.history.keys())
    log_training_progress(f"Dostępne metryki: {available_metrics}")
    
    try:
        # Wykres dokładności i straty
        plt.figure(figsize=(15, 10))
        
        # Wykres dokładności
        plt.subplot(2, 2, 1)
        plt.plot(history.history['accuracy'], label='Trening')
        plt.plot(history.history['val_accuracy'], label='Walidacja')
        plt.title(f'{title} - Dokładność')
        plt.xlabel('Epoka')
        plt.ylabel('Dokładność')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Wykres straty
        plt.subplot(2, 2, 2)
        plt.plot(history.history['loss'], label='Trening')
        plt.plot(history.history['val_loss'], label='Walidacja')
        plt.title(f'{title} - Funkcja straty')
        plt.xlabel('Epoka')
        plt.ylabel('Strata')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Wykresy dla precyzji i czułości, jeśli są dostępne
        precision_key = next((k for k in available_metrics if 'precision' in k.lower() and not k.startswith('val_')), None)
        val_precision_key = next((k for k in available_metrics if 'precision' in k.lower() and k.startswith('val_')), None)
        recall_key = next((k for k in available_metrics if 'recall' in k.lower() and not k.startswith('val_')), None)
        val_recall_key = next((k for k in available_metrics if 'recall' in k.lower() and k.startswith('val_')), None)
        
        if precision_key and val_precision_key:
            plt.subplot(2, 2, 3)
            plt.plot(history.history[precision_key], label='Trening')
            plt.plot(history.history[val_precision_key], label='Walidacja')
            plt.title(f'{title} - Precyzja')
            plt.xlabel('Epoka')
            plt.ylabel('Precyzja')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        if recall_key and val_recall_key:
            plt.subplot(2, 2, 4)
            plt.plot(history.history[recall_key], label='Trening')
            plt.plot(history.history[val_recall_key], label='Walidacja')
            plt.title(f'{title} - Czułość')
            plt.xlabel('Epoka')
            plt.ylabel('Czułość')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        
        # Dodatkowy wykres dla AUC, jeśli dostępny
        auc_key = next((k for k in available_metrics if 'auc' in k.lower() and not k.startswith('val_')), None)
        val_auc_key = next((k for k in available_metrics if 'auc' in k.lower() and k.startswith('val_')), None)
        
        if auc_key and val_auc_key:
            plt.figure(figsize=(10, 6))
            plt.plot(history.history[auc_key], label='Trening')
            plt.plot(history.history[val_auc_key], label='Walidacja')
            plt.title(f'{title} - AUC')
            plt.xlabel('Epoka')
            plt.ylabel('AUC')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(filename.replace('.png', '_auc.png'))
            plt.show()
        
    except Exception as e:
        log_training_progress(f"Błąd podczas tworzenia wykresów: {e}")
        traceback.print_exc()

# Funkcja do ewaluacji modelu
def evaluate_model(model, test_generator, class_names):
    """Ocenia model i generuje raporty wydajności"""
    
    try:
        # Predykcje na zbiorze testowym
        log_training_progress("Wykonywanie predykcji na zbiorze testowym...")
        test_generator.reset()
        predictions = model.predict(test_generator, verbose=1)
        
        # Dla klasyfikacji binarnej
        if model.layers[-1].units == 1:
            y_pred = (predictions > 0.5).astype(int).flatten()
            y_true = test_generator.classes
            
            # Raport klasyfikacji
            log_training_progress("\nRaport klasyfikacji:")
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            print(classification_report(y_true, y_pred, target_names=class_names))
            
            # Zapisz raport do pliku
            with open('classification_report.txt', 'w', encoding='utf-8') as f:
                f.write(f"Model: {MODEL_NAME}\n")
                f.write(f"Data: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(classification_report(y_true, y_pred, target_names=class_names))
            
            # Zapisz metryki do pliku CSV
            with open('model_metrics.csv', 'w', encoding='utf-8') as f:
                f.write("Class,Precision,Recall,F1-Score,Support\n")
                for cls in class_names:
                    f.write(f"{cls},{report[cls]['precision']:.4f},{report[cls]['recall']:.4f},{report[cls]['f1-score']:.4f},{report[cls]['support']}\n")
                f.write(f"accuracy,,,,{report['accuracy']:.4f}\n")
                f.write(f"macro avg,{report['macro avg']['precision']:.4f},{report['macro avg']['recall']:.4f},{report['macro avg']['f1-score']:.4f},{report['macro avg']['support']}\n")
                f.write(f"weighted avg,{report['weighted avg']['precision']:.4f},{report['weighted avg']['recall']:.4f},{report['weighted avg']['f1-score']:.4f},{report['weighted avg']['support']}\n")
            
            # Macierz pomyłek
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Macierz pomyłek')
            plt.colorbar()
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
            
            # Dodanie wartości do macierzy
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('Prawdziwa etykieta')
            plt.xlabel('Przewidziana etykieta')
            plt.savefig('confusion_matrix.png')
            plt.show()
            
            try:
                # Wykres ROC
                fpr, tpr, _ = roc_curve(y_true, predictions)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(10, 8))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC)')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                plt.savefig('roc_curve.png')
                plt.show()
                
                # Wykres Precision-Recall
                precision, recall, _ = precision_recall_curve(y_true, predictions)
                avg_precision = average_precision_score(y_true, predictions)
                
                plt.figure(figsize=(10, 8))
                plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.ylim([0.0, 1.05])
                plt.xlim([0.0, 1.0])
                plt.title('Precision-Recall Curve')
                plt.legend(loc="lower left")
                plt.grid(True, alpha=0.3)
                plt.savefig('precision_recall_curve.png')
                plt.show()
                
            except Exception as e:
                log_training_progress(f"Błąd podczas tworzenia wykresów ROC lub PR: {e}")
                traceback.print_exc()
        
        # Dla klasyfikacji wieloklasowej
        else:
            y_pred = np.argmax(predictions, axis=1)
            y_true = test_generator.classes
            
            # Raport klasyfikacji
            log_training_progress("\nRaport klasyfikacji:")
            print(classification_report(y_true, y_pred, target_names=class_names))
            
            # Macierz pomyłek
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(12, 10))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Macierz pomyłek')
            plt.colorbar()
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
            
            # Dodanie wartości do macierzy
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('Prawdziwa etykieta')
            plt.xlabel('Przewidziana etykieta')
            plt.savefig('confusion_matrix.png')
            plt.show()
    
    except Exception as e:
        log_training_progress(f"Błąd podczas ewaluacji modelu: {e}")
        traceback.print_exc()

# Funkcja pomocnicza do utworzenia bezpiecznych callbacków
def create_safe_callbacks(model, checkpoint_path, monitor='val_loss', mode='min', patience_es=12, patience_lr=5, min_lr=1e-7):
    """Tworzy callbacki z obsługą błędów i zabezpieczeniami"""
    try:
        # Utworzenie katalogu dla logów TensorBoard
        log_dir = os.path.join(LOGS_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        
        callbacks = []
        
        # Early Stopping z większą cierpliwością dla większego zbioru danych
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=patience_es,  # Większa cierpliwość dla większego zbioru
            restore_best_weights=True,
            verbose=1,
            mode=mode
        )
        callbacks.append(early_stopping)
        
        # ReduceLROnPlateau z większą cierpliwością
        reduce_lr = ReduceLROnPlateau(
            monitor=monitor,
            factor=0.3,  # Bardziej agresywne zmniejszanie LR dla większego zbioru
            patience=patience_lr,
            min_lr=min_lr,
            verbose=1,
            mode=mode
        )
        callbacks.append(reduce_lr)
        
        # ModelCheckpoint z zapisem najlepszego modelu
        model_checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,  # Zapisuj cały model
            verbose=1,
            mode=mode
        )
        callbacks.append(model_checkpoint)
        
        # ModelCheckpoint z zapisem ostatniego modelu (zabezpieczenie)
        last_checkpoint = ModelCheckpoint(
            checkpoint_path.replace(".keras", "_last.keras"),
            save_best_only=False,
            save_weights_only=False,
            verbose=0
        )
        callbacks.append(last_checkpoint)
        
        # Zabezpieczenie - dodatkowy zapis co 5 epok
        epoch_checkpoint = ModelCheckpoint(
            os.path.join(CHECKPOINTS_DIR, "epoch_{epoch:02d}.keras"),
            save_weights_only=False,
            save_freq=5 * (900 // BATCH_SIZE),  # Co 5 epok
            verbose=0
        )
        callbacks.append(epoch_checkpoint)
        
        # TensorBoard dla wizualizacji
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=0
        )
        callbacks.append(tensorboard)
        
        # Capture the model in the outer scope for use in the lambda
        # Callback do zapisywania bieżącego LR
        lr_logger = LambdaCallback(
            on_epoch_end=lambda epoch, logs: logs.update({'learning_rate': float(tf.keras.backend.get_value(model.optimizer.lr))})
        )
        callbacks.append(lr_logger)
        
        # Callback informacyjny - wyświetla informacje o epoce
        epoch_info = LambdaCallback(
            on_epoch_begin=lambda epoch, logs: log_training_progress(f"Rozpoczęcie epoki {epoch+1}/{EPOCHS}"),
            on_epoch_end=lambda epoch, logs: log_training_progress(
                f"Epoka {epoch+1}/{EPOCHS}: "
                f"loss={logs.get('loss', 0):.4f}, val_loss={logs.get('val_loss', 0):.4f}, "
                f"acc={logs.get('accuracy', 0):.4f}, val_acc={logs.get('val_accuracy', 0):.4f}"
            )
        )
        callbacks.append(epoch_info)
        
        return callbacks
    
    except Exception as e:
        log_training_progress(f"Błąd podczas tworzenia callbacków: {e}")
        traceback.print_exc()
        
        # Zwróć minimalne callbacki, aby trening mógł kontynuować
        return [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
        ]

# NOWE FUNKCJE DO ANALIZY OBRAZU I OPISÓW DIAGNOSTYCZNYCH
# ======================================================

def analyze_image_features(image_path, image_size):
    """
    Analizuje cechy obrazu USG, które mogą być przydatne do opisu diagnostycznego
    """
    try:
        # Wczytanie obrazu
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size, color_mode='grayscale')
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalizacja
        
        # Analiza intensywności
        mean_intensity = np.mean(img_array)
        std_intensity = np.std(img_array)
        
        # Analiza kontrastu
        contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0
        
        # Analiza jednorodności tkanek
        histogram, _ = np.histogram(img_array, bins=10, range=(0, 1))
        histogram_norm = histogram / np.sum(histogram)
        entropy = -np.sum(histogram_norm * np.log2(histogram_norm + 1e-10))
        
        # Detekcja krawędzi (do wykrywania struktur)
        edges = ndimage.sobel(img_array[:,:,0])
        edge_magnitude = np.mean(np.abs(edges))
        
        # Określenie regionów o dużej intensywności (potencjalne struktury płynu)
        fluid_threshold = 0.7
        potential_fluid = np.sum(img_array > fluid_threshold) / img_array.size
        
        # Określenie regionów o niskiej intensywności (potencjalne struktury tkankowe)
        tissue_threshold = 0.3
        potential_tissue = np.sum(img_array < tissue_threshold) / img_array.size
        
        return {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'contrast': contrast,
            'entropy': entropy,
            'edge_magnitude': edge_magnitude,
            'potential_fluid_ratio': potential_fluid,
            'potential_tissue_ratio': potential_tissue
        }
    except Exception as e:
        print(f"Błąd analizy obrazu: {e}")
        traceback.print_exc()
        return {
            'mean_intensity': 0.5,
            'std_intensity': 0.2,
            'contrast': 0.4,
            'entropy': 4.0,
            'edge_magnitude': 0.3,
            'potential_fluid_ratio': 0.2,
            'potential_tissue_ratio': 0.3
        }

def generate_description(predicted_class, confidence, image_path=None, additional_info=None):
    """
    Generuje opis diagnostyczny na podstawie klasy predykcji, pewności modelu i analizy obrazu
    """
    # Analiza obrazu, jeśli dostępna ścieżka
    image_features = None
    if image_path:
        try:
            image_features = analyze_image_features(image_path, IMAGE_SIZE)
        except Exception as e:
            print(f"Błąd podczas analizy obrazu: {e}")
    
    current_date = datetime.datetime.now().strftime("%d.%m.%Y")
    
    # Określenie dnia cyklu (przykładowe obliczenie - należy dostosować)
    # W rzeczywistym systemie możesz pobierać tę informację z bazy danych lub od użytkownika
    estimated_day = "nieznany"  # Domyślnie nieznany
    
    # Jeśli dostępne są dodatkowe informacje
    if additional_info:
        if 'estimated_day' in additional_info and additional_info['estimated_day']:
            estimated_day = additional_info['estimated_day']
    
    # Jeśli nazwa pliku zawiera informację o dniu, możesz ją wyekstrahować
    if image_path:
        filename = os.path.basename(image_path)
        day_match = re.search(r'_d(\d+)_', filename)
        if day_match:
            estimated_day = day_match.group(1)
    
    # Generowanie opisu w zależności od predykcji
    if predicted_class == "pregnant":
        # Opis dla ciąży z uwzględnieniem pewności modelu
        if confidence > 0.95:
            description = (
                f"Badanie USG z dnia {current_date} wykazuje obraz jednoznacznie wskazujący na ciążę u klaczy. "
                f"Widoczne są charakterystyczne struktury pęcherzyka ciążowego z dobrze zarysowanymi granicami. "
                f"Szacowany dzień ciąży: około {estimated_day} dni (weryfikacja kliniczna wskazana). "
                f"Rozwój zarodka przebiega prawidłowo, bez widocznych nieprawidłowości. "
                f"Zalecana kontrola rozwoju ciąży za 7-14 dni."
            )
        elif confidence > 0.85:
            description = (
                f"Badanie USG z dnia {current_date} wskazuje z dużym prawdopodobieństwem na ciążę u klaczy. "
                f"Widoczny jest pęcherzyk ciążowy. Szacowany dzień ciąży: około {estimated_day} dni. "
                f"Obraz wymaga potwierdzenia w badaniu kontrolnym. "
                f"Zalecana kontrola za 7-10 dni dla potwierdzenia prawidłowego rozwoju ciąży."
            )
        else:
            description = (
                f"Badanie USG z dnia {current_date} sugeruje możliwą ciążę u klaczy, "
                f"jednak obraz nie jest jednoznaczny (pewność predykcji: {confidence*100:.1f}%). "
                f"Widoczne są struktury mogące odpowiadać wczesnej ciąży. "
                f"Zalecane jest pilne badanie kontrolne za 5-7 dni dla weryfikacji. "
                f"Należy zachować ostrożność diagnostyczną."
            )
    else:  # not_pregnant
        if confidence > 0.95:
            description = (
                f"Badanie USG z dnia {current_date} jednoznacznie nie wykazuje cech ciąży u klaczy. "
                f"Obraz macicy prawidłowy, bez widocznych struktur ciążowych. "
                f"Zalecana kontrola cyklu rujowego i powtórzenie badania w odpowiednim momencie "
                f"cyklu dla weryfikacji."
            )
        elif confidence > 0.85:
            description = (
                f"Badanie USG z dnia {current_date} nie wykazuje jednoznacznych cech ciąży u klaczy. "
                f"Obraz macicy bez wyraźnych struktur ciążowych. Możliwe wczesne stadium cyklu rujowego. "
                f"Zalecana obserwacja objawów rui i powtórzenie badania w odpowiednim czasie."
            )
        else:
            description = (
                f"Badanie USG z dnia {current_date} sugeruje brak ciąży, "
                f"jednak pewność predykcji jest umiarkowana ({confidence*100:.1f}%). "
                f"Obraz wymaga ostrożnej interpretacji. Zalecane powtórzenie badania za 5-7 dni. "
                f"Wskazana obserwacja objawów klinicznych klaczy."
            )
    
    # Dodanie informacji o jakości obrazu, jeśli dostępne
    if image_features:
        if image_features['contrast'] < 0.3:
            description += (
                f"\n\nUWAGA: Obraz USG charakteryzuje się niskim kontrastem ({image_features['contrast']:.2f}), "
                f"co może utrudniać diagnostykę. Zalecane powtórzenie badania z lepszymi parametrami aparatu."
            )
        
        if image_features['entropy'] > 5.0:
            description += (
                f"\n\nUWAGA: Obraz USG zawiera znaczny szum (entropia: {image_features['entropy']:.2f}), "
                f"zalecane powtórzenie badania lub użycie filtrów redukcji szumów."
            )
            
        # Detekcja potencjalnych struktur płynowych
        if predicted_class == "pregnant" and image_features['potential_fluid_ratio'] > 0.3:
            description += (
                f"\n\nW obrazie USG widoczne są wyraźne struktury płynowe "
                f"charakterystyczne dla pęcherzyka ciążowego."
            )
    
    # Dodanie ogólnych zaleceń
    description += (
        f"\n\nZALECENIA: "
        f"\n1. Utrzymanie odpowiedniego żywienia i suplementacji klaczy."
        f"\n2. Regularna kontrola weterynaryjna."
        f"\n3. Monitorowanie ogólnego stanu zdrowia klaczy."
    )
    
    # Dodanie informacji o klaczy z dodatkowych informacji
    if additional_info:
        mare_info = ""
        if 'klacz_name' in additional_info and additional_info['klacz_name']:
            mare_info += f"\nImię klaczy: {additional_info['klacz_name']}"
        if 'klacz_age' in additional_info and additional_info['klacz_age']:
            mare_info += f"\nWiek klaczy: {additional_info['klacz_age']} lat"
            
        if mare_info:
            description = f"INFORMACJE O KLACZY:{mare_info}\n\n" + description
    
    # Dodanie zastrzeżenia
    description += (
        f"\n\nUWAGA: Powyższy opis został wygenerowany automatycznie przez system AI "
        f"i wymaga weryfikacji przez lekarza weterynarii. Pewność predykcji: {confidence*100:.2f}%."
    )
    
    return description

def predict_image_with_description(image_path, model, class_names):
    """
    Funkcja do predykcji pojedynczego obrazu z generowaniem opisu diagnostycznego
    """
    try:
        # Ładowanie i przygotowanie obrazu
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = applications.inception_v3.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predykcja
        start_time = time.time()
        prediction = model.predict(img_array)[0]
        inference_time = time.time() - start_time
        
        # Dla klasyfikacji binarnej
        if isinstance(prediction, (float, np.float32, np.float64)) or len(prediction.shape) == 0 or prediction.shape[0] == 1:
            confidence = float(prediction) if len(prediction.shape) == 0 else float(prediction[0])
            predicted_class = class_names[1] if confidence > 0.5 else class_names[0]
            
            # Generowanie opisu na podstawie predykcji
            description = generate_description(predicted_class, confidence, image_path)
            
            print(f"Predykcja: {predicted_class} z pewnością {confidence*100:.2f}%")
            print(f"Czas wnioskowania: {inference_time*1000:.2f} ms")
            print(f"\nOpis diagnostyczny:\n{description}")
        else:
            # Dla klasyfikacji wieloklasowej (jeśli byłaby potrzebna w przyszłości)
            predicted_class_idx = np.argmax(prediction)
            confidence = prediction[predicted_class_idx]
            predicted_class = class_names[predicted_class_idx]
            
            # Generowanie opisu na podstawie predykcji
            description = generate_description(predicted_class, confidence, image_path)
            
            print(f"Predykcja: {predicted_class} z pewnością {confidence*100:.2f}%")
            print(f"Czas wnioskowania: {inference_time*1000:.2f} ms")
            print(f"\nOpis diagnostyczny:\n{description}")
        
        # Wyświetlenie obrazu z predykcją i fragmentem opisu
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        
        # Dodanie tytułu z predykcją
        plt.title(f"Predykcja: {predicted_class} ({confidence*100:.2f}%)\nCzas: {inference_time*1000:.1f} ms")
        
        # Dodanie opisu jako tekst pod obrazem
        short_desc = description.split('\n')[0]  # Pierwszy wiersz opisu
        plt.figtext(0.5, 0.01, short_desc, wrap=True, horizontalalignment='center', fontsize=12)
        
        plt.axis('off')
        plt.tight_layout()
        
        # Zapisanie obrazu z opisem
        result_filename = os.path.join(REPORTS_DIR, f"prediction_result_{os.path.basename(image_path)}.png")
        plt.savefig(result_filename, bbox_inches='tight')
        plt.show()
        
        # Zapisanie pełnego raportu do pliku tekstowego
        report_filename = os.path.join(REPORTS_DIR, f"usg_report_{os.path.basename(image_path).split('.')[0]}.txt")
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(f"RAPORT DIAGNOSTYCZNY USG\n")
            f.write(f"=======================\n\n")
            f.write(f"Data badania: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Plik obrazu: {image_path}\n")
            f.write(f"Predykcja modelu: {predicted_class}\n")
            f.write(f"Pewność predykcji: {confidence*100:.2f}%\n")
            f.write(f"Czas wnioskowania: {inference_time*1000:.2f} ms\n\n")
            f.write(f"OPIS DIAGNOSTYCZNY:\n")
            f.write(f"{description}\n\n")
            f.write(f"=======================\n")
            f.write(f"Raport wygenerowany automatycznie przez system AI.\n")
            f.write(f"Zalecana weryfikacja przez specjalistę weterynarii.\n")
        
        print(f"Zapisano pełny raport do pliku: {report_filename}")
        
        return predicted_class, confidence, description
    
    except Exception as e:
        print(f"Błąd podczas wykonywania predykcji: {e}")
        traceback.print_exc()
        return None, None, None

def generate_pdf_report(image_path, predicted_class, confidence, description, additional_info=None):
    """
    Generuje raport PDF z wynikami analizy USG
    """
    # Nazwa pliku raportu
    report_filename = os.path.join(REPORTS_DIR, f"usg_report_{os.path.basename(image_path).split('.')[0]}.pdf")
    
    # Utworzenie dokumentu PDF
    doc = SimpleDocTemplate(report_filename, pagesize=letter)
    story = []
    
    # Style
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='MyHeading1', fontSize=16, spaceAfter=12))
    styles.add(ParagraphStyle(name='MyHeading2', fontSize=14, spaceAfter=8))
    styles.add(ParagraphStyle(name='MyNormal', fontSize=12, spaceAfter=8))
    
    # Tytuł
    story.append(Paragraph(f"RAPORT DIAGNOSTYCZNY USG", styles['MyHeading1']))
    story.append(Spacer(1, 0.2*inch))
    
    # Informacje o badaniu
    story.append(Paragraph(f"Data badania: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['MyNormal']))
    
    if additional_info:
        if additional_info.get('klacz_name'):
            story.append(Paragraph(f"Imię klaczy: {additional_info['klacz_name']}", styles['MyNormal']))
        if additional_info.get('klacz_age'):
            story.append(Paragraph(f"Wiek klaczy: {additional_info['klacz_age']} lat", styles['Normal']))
        if additional_info.get('estimated_day'):
            story.append(Paragraph(f"Szacowany dzień cyklu/ciąży: {additional_info['estimated_day']}", styles['Normal']))
    
    story.append(Paragraph(f"Plik obrazu: {os.path.basename(image_path)}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Wynik analizy
    story.append(Paragraph("WYNIK ANALIZY", styles['MyHeading2']))
    story.append(Paragraph(f"Predykcja modelu: <b>{predicted_class}</b>", styles['Normal']))
    story.append(Paragraph(f"Pewność predykcji: {confidence*100:.2f}%", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Dodanie obrazu USG
    img = ReportLabImage(image_path, width=5*inch, height=4*inch)
    story.append(img)
    story.append(Spacer(1, 0.2*inch))
    
    # Opis diagnostyczny
    story.append(Paragraph("OPIS DIAGNOSTYCZNY", styles['Heading2']))
    # Rozbij opis na akapity dla lepszej czytelności
    for para in description.split('\n\n'):
        story.append(Paragraph(para, styles['Normal']))
    
    # Zastrzeżenie
    story.append(Spacer(1, 0.3*inch))
    disclaimer = "Raport wygenerowany automatycznie przez system AI. Zalecana weryfikacja przez specjalistę weterynarii."
    story.append(Paragraph(disclaimer, styles['Normal']))
    
    # Wygeneruj PDF
    doc.build(story)
    
    print(f"Wygenerowano raport PDF: {report_filename}")
    return report_filename

def predict_with_ui(model, class_names):
    """
    Funkcja interfejsu użytkownika do predykcji z opisem
    """
    # Konfiguracja matplotlib w trybie nie-interaktywnym dla tkinter
    matplotlib.use('Agg')
    
    # Utworzenie okna
    root = tk.Tk()
    root.title("VETEYE - System predykcji ciąży klaczy na podstawie USG -ALK.BIZNES.AI.G12.G2")
    root.geometry("1100x800")
    
    # Zmienne
    selected_image_path = tk.StringVar()
    klacz_name = tk.StringVar()
    klacz_age = tk.StringVar()
    estimated_day = tk.StringVar()
    
    # Funkcje
    def select_image():
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            selected_image_path.set(path)
            # Wczytaj i wyświetl obraz
            img = Image.open(path)
            img = img.resize((400, 400), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            image_label.configure(image=img_tk)
            image_label.image = img_tk
    
    def run_prediction():
        image_path = selected_image_path.get()
        if not image_path:
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, "Proszę wybrać obraz najpierw!")
            return
        
        # Dodanie informacji o klaczy do opisu
        additional_info = {
            'klacz_name': klacz_name.get(),
            'klacz_age': klacz_age.get(),
            'estimated_day': estimated_day.get()
        }
        
        # Wykonanie predykcji
        try:
            # Ładowanie i przygotowanie obrazu
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = applications.inception_v3.preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predykcja
            start_time = time.time()
            prediction = model.predict(img_array)[0]
            inference_time = time.time() - start_time
            
            # Dla klasyfikacji binarnej
            confidence = float(prediction) if len(prediction.shape) == 0 else float(prediction[0])
            predicted_class = class_names[1] if confidence > 0.5 else class_names[0]
            
            # Generowanie opisu na podstawie predykcji
            description = generate_description(predicted_class, confidence, image_path, additional_info)
            
            # Wyświetlenie wyników
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"WYNIK PREDYKCJI:\n")
            result_text.insert(tk.END, f"Klasa: {predicted_class}\n")
            result_text.insert(tk.END, f"Pewność: {confidence*100:.2f}%\n")
            result_text.insert(tk.END, f"Czas: {inference_time*1000:.2f} ms\n\n")
            result_text.insert(tk.END, f"OPIS DIAGNOSTYCZNY:\n")
            result_text.insert(tk.END, description)
            
            # Wygenerowanie PDF
            pdf_path = generate_pdf_report(image_path, predicted_class, confidence, description, additional_info)
            result_text.insert(tk.END, f"\n\nWygenerowano raport PDF: {pdf_path}")
            
        except Exception as e:
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"Błąd podczas analizy obrazu: {e}")
            traceback.print_exc()
    
    # Interfejs
    main_frame = Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    # Panel lewy - obraz
    left_frame = Frame(main_frame)
    left_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH)
    
    select_button = Button(left_frame, text="Wybierz obraz USG", command=select_image, width=20)
    select_button.pack(pady=10)
    
    image_label = Label(left_frame, text="Obraz pojawi się tutaj", width=400, height=400, bg="lightgray")
    image_label.pack(pady=10)
    
    # Panel środkowy - dane klaczy
    middle_frame = Frame(main_frame)
    middle_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH)
    
    Label(middle_frame, text="Imię klaczy:").pack(anchor=tk.W, pady=5)
    Entry(middle_frame, textvariable=klacz_name, width=30).pack(pady=5)
    
    Label(middle_frame, text="Wiek klaczy (lata):").pack(anchor=tk.W, pady=5)
    Entry(middle_frame, textvariable=klacz_age, width=30).pack(pady=5)
    
    Label(middle_frame, text="Szacowany dzień cyklu/ciąży:").pack(anchor=tk.W, pady=5)
    Entry(middle_frame, textvariable=estimated_day, width=30).pack(pady=5)
    
    analyze_button = Button(middle_frame, text="Analizuj obraz", command=run_prediction, width=20, bg="lightblue")
    analyze_button.pack(pady=20)
    
    # Panel prawy - wyniki
    right_frame = Frame(main_frame)
    right_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
    
    Label(right_frame, text="Wynik analizy:").pack(anchor=tk.W, pady=5)
    
    result_frame = Frame(right_frame)
    result_frame.pack(fill=tk.BOTH, expand=True)
    
    result_text = Text(result_frame, wrap=tk.WORD, width=50, height=25)
    result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    scrollbar = Scrollbar(result_frame, command=result_text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    result_text.config(yscrollcommand=scrollbar.set)
    
    return root
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",   action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    CLASS_NAMES = ["not_pregnant", "pregnant"]

    # ←  wczytanie gotowego modelu
    model = tf.keras.models.load_model(
        r"checkpoints\usg_pregnancy_model_inceptionv3_380x380_fina.keras",
        compile=False
    )

    if args.train:
        # tu wszystko związane z load_data(), create_transfer_model() itd.
        ...
    elif args.analyze:
        root = predict_with_ui(model, CLASS_NAMES)
        root.mainloop()          # ← bez tego okno nigdy się nie pokaże
    
