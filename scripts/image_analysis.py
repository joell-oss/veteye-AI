# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import numpy as np
import tensorflow as tf
from scipy import ndimage
import cv2
import traceback
import re
import os
from skimage.feature import graycomatrix, graycoprops
from logging_utils import log_info, log_error
from config import FEATURE_EXTRACTION_PARAMS

def analyze_image_features(image_path, image_size):
    """
    Analizuje cechy obrazu USG, które mogą być przydatne do opisu diagnostycznego.
    """
    try:
        # Wczytanie obrazu w skali szarości
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size, color_mode='grayscale')
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalizacja

        # Analiza intensywności
        mean_intensity = np.mean(img_array)
        std_intensity = np.std(img_array)
        contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0

        # Entropia
        histogram, _ = np.histogram(img_array, bins=10, range=(0, 1))
        histogram_norm = histogram / np.sum(histogram)
        entropy = -np.sum(histogram_norm * np.log2(histogram_norm + 1e-10))

        # Procent pikseli o wysokiej intensywności (płyn)
        fluid_threshold = 0.7
        potential_fluid = np.sum(img_array > fluid_threshold) / img_array.size

        # Procent pikseli o niskiej intensywności (tkanka)
        tissue_threshold = 0.3
        potential_tissue = np.sum(img_array < tissue_threshold) / img_array.size

        return {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'contrast': contrast,
            'entropy': entropy,
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
            'potential_fluid_ratio': 0.2,
            'potential_tissue_ratio': 0.3
        }

def load_and_preprocess_image_for_analysis(image_path, target_size=(380, 380)):
    """Ładuje i przetwarza obraz USG do analizy cech"""
    try:
        # Załadowanie obrazu w skali szarości
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Nie można załadować obrazu: {image_path}")
            
        # Zmiana rozmiaru obrazu
        img_resized = cv2.resize(img, target_size)
        
        # Normalizacja wartości pikseli do zakresu [0, 1]
        img_norm = img_resized / 255.0
        
        return img_norm, img_resized
    
    except Exception as e:
        log_error(f"Błąd podczas ładowania obrazu do analizy: {e}")
        return None, None

def extract_image_features(image_array, params=FEATURE_EXTRACTION_PARAMS):
    """Ekstrahuje cechy obrazu USG istotne dla określenia ciąży i dnia ciąży"""
    
    try:
        # Parametry analizy
        num_regions = params['num_regions']
        edge_detection_threshold = params['edge_detection_threshold']
        fluid_threshold = params['fluid_threshold']
        tissue_threshold = params['tissue_threshold']
        
        # 1. Analiza intensywności globalnej
        mean_intensity = np.mean(image_array)
        std_intensity = np.std(image_array)
        min_intensity = np.min(image_array)
        max_intensity = np.max(image_array)
        
        # 2. Kontrast
        contrast = (max_intensity - min_intensity) / (max_intensity + min_intensity + 1e-8)
        
        # 3. Analiza histogramu
        histogram, bins = np.histogram(image_array, bins=10, range=(0, 1))
        histogram_norm = histogram / np.sum(histogram)
        entropy = -np.sum(histogram_norm * np.log2(histogram_norm + 1e-10))
        
        # 4. Detekcja krawędzi
        edges = ndimage.sobel(image_array)
        edge_magnitude = np.mean(np.abs(edges))
        edge_ratio = np.sum(np.abs(edges) > edge_detection_threshold) / edges.size
        
        # 5. Rozpoznanie struktur płynowych i tkankowych
        potential_fluid = np.sum(image_array > fluid_threshold) / image_array.size
        potential_tissue = np.sum(image_array < tissue_threshold) / image_array.size
        
        # 6. Analiza tekstury
        # Przekształcenie obrazu do uint8 dla GLCM
        img_uint8 = (image_array * 255).astype(np.uint8)
        # Obliczenie macierzy GLCM
        distances = [1, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(img_uint8, distances=distances, angles=angles, 
                           levels=8, symmetric=True, normed=True)
        # Obliczenie właściwości GLCM
        glcm_contrast = np.mean(graycoprops(glcm, 'contrast'))
        glcm_dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
        glcm_homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
        glcm_energy = np.mean(graycoprops(glcm, 'energy'))
        glcm_correlation = np.mean(graycoprops(glcm, 'correlation'))
        
        # 7. Analiza regionów
        h, w = image_array.shape
        region_features = []
        
        # Podziel obraz na regiony i analizuj każdy
        region_h = h // int(np.sqrt(num_regions))
        region_w = w // int(np.sqrt(num_regions))
        
        for i in range(0, h, region_h):
            for j in range(0, w, region_w):
                if i + region_h <= h and j + region_w <= w:
                    region = image_array[i:i+region_h, j:j+region_w]
                    region_features.append({
                        'mean': np.mean(region),
                        'std': np.std(region),
                        'edge_magnitude': np.mean(np.abs(ndimage.sobel(region))),
                        'fluid_ratio': np.sum(region > fluid_threshold) / region.size,
                    })
        
        # Utwórz słownik z cechami
        features = {
            'basic': {
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity),
                'min_intensity': float(min_intensity),
                'max_intensity': float(max_intensity),
                'contrast': float(contrast),
                'entropy': float(entropy)
            },
            'edges': {
                'edge_magnitude': float(edge_magnitude),
                'edge_ratio': float(edge_ratio)
            },
            'structures': {
                'potential_fluid_ratio': float(potential_fluid),
                'potential_tissue_ratio': float(potential_tissue)
            },
            'texture': {
                'glcm_contrast': float(glcm_contrast),
                'glcm_dissimilarity': float(glcm_dissimilarity),
                'glcm_homogeneity': float(glcm_homogeneity),
                'glcm_energy': float(glcm_energy),
                'glcm_correlation': float(glcm_correlation)
            },
            'regions': region_features
        }
        
        # Spłaszczony słownik dla łatwiejszego wykorzystania w uczeniu maszynowym
        flat_features = {
            'mean_intensity': float(mean_intensity),
            'std_intensity': float(std_intensity),
            'contrast': float(contrast),
            'entropy': float(entropy),
            'edge_magnitude': float(edge_magnitude),
            'edge_ratio': float(edge_ratio),
            'potential_fluid_ratio': float(potential_fluid),
            'potential_tissue_ratio': float(potential_tissue),
            'glcm_contrast': float(glcm_contrast),
            'glcm_homogeneity': float(glcm_homogeneity),
            'glcm_energy': float(glcm_energy),
            'glcm_correlation': float(glcm_correlation)
        }
        
        return features, flat_features
    
    except Exception as e:
        log_error(f"Błąd podczas ekstrahowania cech obrazu: {e}")
        return None, None

def visualize_image_analysis(image_array, features, output_path=None):
    """Tworzy wizualizację analizy obrazu z podświetlonymi obszarami"""
    
    try:
        # Przekształć na obraz kolorowy
        img_color = cv2.cvtColor((image_array * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # Detekcja krawędzi
        edges = ndimage.sobel(image_array)
        edge_mask = np.abs(edges) > FEATURE_EXTRACTION_PARAMS['edge_detection_threshold']
        
        # Maska płynu
        fluid_mask = image_array > FEATURE_EXTRACTION_PARAMS['fluid_threshold']
        
        # Maska tkanki
        tissue_mask = image_array < FEATURE_EXTRACTION_PARAMS['tissue_threshold']
        
        # Nakładanie masek na obraz
        img_color[edge_mask] = [255, 0, 0]  # Krawędzie na czerwono
        img_color[fluid_mask] = [0, 0, 255]  # Płyn na niebiesko
        img_color[tissue_mask] = [0, 255, 0]  # Tkanka na zielono
        
        # Dodaj tekst z podstawowymi cechami
        h, w = image_array.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Przygotuj obraz z tekstem
        text_img = np.zeros((h + 150, w, 3), dtype=np.uint8) + 255
        text_img[:h, :w] = img_color
        
        # Dodaj opisy
        cv2.putText(text_img, f"Mean intensity: {features['basic']['mean_intensity']:.3f}", (10, h + 30), font, 0.5, (0, 0, 0), 1)
        cv2.putText(text_img, f"Contrast: {features['basic']['contrast']:.3f}", (10, h + 60), font, 0.5, (0, 0, 0), 1)
        cv2.putText(text_img, f"Edge magnitude: {features['edges']['edge_magnitude']:.3f}", (10, h + 90), font, 0.5, (0, 0, 0), 1)
        cv2.putText(text_img, f"Fluid ratio: {features['structures']['potential_fluid_ratio']:.3f}", (10, h + 120), font, 0.5, (0, 0, 0), 1)
        
        cv2.putText(text_img, f"Krawędzie: Czerwony", (w//2, h + 30), font, 0.5, (0, 0, 255), 1)
        cv2.putText(text_img, f"Płyn: Niebieski", (w//2, h + 60), font, 0.5, (255, 0, 0), 1)
        cv2.putText(text_img, f"Tkanka: Zielony", (w//2, h + 90), font, 0.5, (0, 255, 0), 1)
        
        # Zapisz wynik
        if output_path:
            cv2.imwrite(output_path, text_img)
        
        return text_img
    
    except Exception as e:
        log_error(f"Błąd podczas wizualizacji analizy obrazu: {e}")
        return None
