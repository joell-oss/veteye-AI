# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import os
import argparse
import shutil
import random
import cv2
import numpy as np
import glob
from logging_utils import setup_logging, log_info, log_error, log_success, log_section
from config import DATA_DIR, TRAIN_DIR, TEST_DIR

def parse_arguments():
    """
    Konfiguruje parametry wiersza poleceń dla przygotowania zestawu danych USG.
    Definiuje interfejs CLI z następującymi opcjami:
    - source (-s): wymagany katalog źródłowy z obrazami USG do przetworzenia
    - dest (-d): katalog docelowy (domyślnie DATA_DIR)
    - split (-p): współczynnik podziału na dane testowe (0.0-1.0, domyślnie 20%)
    - structure (-t): typ organizacji danych ('binary' lub 'days')
    - augment (-a): flaga włączająca augmentację danych (opcjonalna)
    Typy struktur danych:
    - 'binary': organizacja ciąża/brak_ciąży (klasyfikacja binarna)
    - 'days': organizacja według dni ciąży (regresja/klasyfikacja wieloklasowa)
    Parametry walidacji:
    - source: obowiązkowy parametr wejściowy
    - split: wartość zmiennoprzecinkowa w zakresie 0.0-1.0
    - structure: ograniczone wybory dla bezpieczeństwa typów
    Umożliwia elastyczną konfigurację procesu przygotowania danych
    bez konieczności modyfikacji kodu źródłowego.
    """
    parser = argparse.ArgumentParser(description="Przygotowanie zestawu danych USG klaczy")
    parser.add_argument("--source", "-s", required=True, help="Katalog źródłowy z obrazami USG")
    parser.add_argument("--dest", "-d", default=DATA_DIR, help="Katalog docelowy na zestaw danych")
    parser.add_argument("--split", "-p", type=float, default=0.2, help="Procent danych testowych (0.0-1.0)")
    parser.add_argument("--structure", "-t", choices=["binary", "days"], default="binary", 
                       help="Struktura danych: 'binary' dla ciąża/brak ciąży, 'days' dla dni ciąży")
    parser.add_argument("--augment", "-a", action="store_true", help="Zastosuj augmentację danych")
    
    return parser.parse_args()

def create_directory_structure(dest_dir, structure="binary"):
    """
    Tworzy hierarchiczną strukturę katalogów dostosowaną do typu zadania uczenia maszynowego.
    Implementuje dwa wzorce organizacji danych:
    Struktura 'binary' (klasyfikacja binarna):
    - Training/pregnant - obrazy z ciążą do treningu
    - Training/not_pregnant - obrazy bez ciąży do treningu  
    - Test/pregnant - obrazy z ciążą do testów
    - Test/not_pregnant - obrazy bez ciąży do testów
    Struktura 'days' (regresja/klasyfikacja wieloklasowa):
    - Training/ - wszystkie obrazy treningowe (etykiety w metadanych)
    - Test/ - wszystkie obrazy testowe (etykiety w metadanych)
    Zastosowanie:
    - 'binary': kompatybilne z ImageDataGenerator.flow_from_directory()
    - 'days': wymaga custom data loader z parsowaniem etykiet z nazw plików
    Automatyczne tworzenie katalogów z exist_ok=True zapewnia 
    bezpieczne wykonanie bez błędów przy istniejących strukturach.
    """
    if structure == "binary":
        # Struktura dla klasyfikacji binarnej (ciąża/brak ciąży)
        os.makedirs(os.path.join(dest_dir, "Training", "pregnant"), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "Training", "not_pregnant"), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "Test", "pregnant"), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "Test", "not_pregnant"), exist_ok=True)
    elif structure == "days":
        # Struktura dla klasyfikacji dni ciąży
        os.makedirs(os.path.join(dest_dir, "Training"), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "Test"), exist_ok=True)

def process_binary_dataset(source_dir, dest_dir, test_split=0.2, apply_augmentation=False, log_file=None):
    """
    Przetwarza zestaw danych obrazowych do klasyfikacji dwuwartościowej (ciąża/brak ciąży).
    Funkcja wczytuje obrazy z katalogów źródłowych, dzieli je na zbiory treningowe 
    i testowe w zadanej proporcji, kopiuje do struktury katalogów docelowych 
    oraz opcjonalnie wykonuje wzbogacanie danych treningowych w celu zwiększenia 
    liczebności zbioru.
    Parametry:
       katalog_źródłowy (str): Ścieżka do katalogu źródłowego zawierającego podkatalogi 
                              'pregnant' i 'not_pregnant' z obrazami
       katalog_docelowy (str): Ścieżka do katalogu docelowego dla struktury Training/Test
       podział_testowy (float): Proporcja danych przeznaczonych na zbiór testowy (domyślnie 0.2)
       zastosuj_wzbogacanie (bool): Czy zastosować wzbogacanie danych treningowych
       plik_logów (str, opcjonalny): Ścieżka do pliku dziennika zdarzeń
    Zwraca:
       bool: Prawda jeśli przetwarzanie zakończyło się pomyślnie, Fałsz w przypadku błędu
    Wyjątki:
       Zapisuje błędy do pliku dziennika jeśli katalogi źródłowe nie istnieją
    """   
    log_section("Przetwarzanie zestawu danych binarnych", log_file)
    
    # Ścieżki katalogów
    pregnant_dir = os.path.join(source_dir, "pregnant")
    not_pregnant_dir = os.path.join(source_dir, "not_pregnant")
    
    # Sprawdź, czy katalogi istnieją
    if not os.path.exists(pregnant_dir) or not os.path.exists(not_pregnant_dir):
        log_error(f"Katalogi źródłowe nie istnieją: {pregnant_dir} lub {not_pregnant_dir}", log_file)
        return False
    
    # Znajdź obrazy
    pregnant_images = glob.glob(os.path.join(pregnant_dir, "*.jpg")) + \
                     glob.glob(os.path.join(pregnant_dir, "*.jpeg")) + \
                     glob.glob(os.path.join(pregnant_dir, "*.png"))
    
    not_pregnant_images = glob.glob(os.path.join(not_pregnant_dir, "*.jpg")) + \
                         glob.glob(os.path.join(not_pregnant_dir, "*.jpeg")) + \
                         glob.glob(os.path.join(not_pregnant_dir, "*.png"))
    
    log_info(f"Znaleziono {len(pregnant_images)} obrazów ciąży", log_file)
    log_info(f"Znaleziono {len(not_pregnant_images)} obrazów bez ciąży", log_file)
    
    # Podziel dane na treningowe i testowe
    random.shuffle(pregnant_images)
    random.shuffle(not_pregnant_images)
    
    # Indeksy podziału
    pregnant_split_idx = int(len(pregnant_images) * (1 - test_split))
    not_pregnant_split_idx = int(len(not_pregnant_images) * (1 - test_split))
    
    # Podział na treningowe i testowe
    pregnant_train = pregnant_images[:pregnant_split_idx]
    pregnant_test = pregnant_images[pregnant_split_idx:]
    
    not_pregnant_train = not_pregnant_images[:not_pregnant_split_idx]
    not_pregnant_test = not_pregnant_images[not_pregnant_split_idx:]
    
    log_info(f"Dane treningowe: {len(pregnant_train)} ciąża, {len(not_pregnant_train)} brak ciąży", log_file)
    log_info(f"Dane testowe: {len(pregnant_test)} ciąża, {len(not_pregnant_test)} brak ciąży", log_file)
    
    # Kopiuj pliki
    copy_images(pregnant_train, os.path.join(dest_dir, "Training", "pregnant"), log_file)
    copy_images(not_pregnant_train, os.path.join(dest_dir, "Training", "not_pregnant"), log_file)
    copy_images(pregnant_test, os.path.join(dest_dir, "Test", "pregnant"), log_file)
    copy_images(not_pregnant_test, os.path.join(dest_dir, "Test", "not_pregnant"), log_file)
    
    # Augmentacja (tylko dla zbioru treningowego)
    if apply_augmentation:
        log_info("Rozpoczynanie augmentacji danych treningowych...", log_file)
        
        # Augmentacja dla klasy 'pregnant'
        augment_images(
            os.path.join(dest_dir, "Training", "pregnant"),
            num_augmentations=max(1, int(5000 / len(pregnant_train))),
            log_file=log_file
        )
        
        # Augmentacja dla klasy 'not_pregnant'
        augment_images(
            os.path.join(dest_dir, "Training", "not_pregnant"),
            num_augmentations=max(1, int(5000 / len(not_pregnant_train))),
            log_file=log_file
        )
    
    log_success("Przetwarzanie danych binarnych zakończone", log_file)
    return True

def process_day_dataset(source_dir, dest_dir, test_split=0.2, apply_augmentation=False, log_file=None):
    """
    Przetwarza zestaw danych obrazowych do klasyfikacji wieloklasowej dni ciąży.
    Funkcja wczytuje obrazy z katalogów numerowanych odpowiadających poszczególnym 
    dniom ciąży, dzieli je na zbiory treningowe i testowe w zadanej proporcji, 
    kopiuje do struktury katalogów docelowych oraz opcjonalnie wykonuje wzbogacanie 
    danych treningowych dla każdej klasy osobno.
    Parametry:
       katalog_źródłowy (str): Ścieżka do katalogu źródłowego zawierającego podkatalogi 
                              numerowane (np. "1", "2", "30") z obrazami dla poszczególnych dni
       katalog_docelowy (str): Ścieżka do katalogu docelowego dla struktury Training/Test
       podział_testowy (float): Proporcja danych przeznaczonych na zbiór testowy (domyślnie 0.2)
       zastosuj_wzbogacanie (bool): Czy zastosować wzbogacanie danych treningowych
       plik_logów (str, opcjonalny): Ścieżka do pliku dziennika zdarzeń
    Zwraca:
       bool: Prawda jeśli przetwarzanie zakończyło się pomyślnie, Fałsz w przypadku błędu
    Uwagi:
       Katalogi źródłowe muszą mieć nazwy będące liczbami całkowitymi reprezentującymi dni ciąży
    Funkcja nie wykonywana w demonstratorze.
    """
    log_section("Przetwarzanie zestawu danych dni ciąży", log_file)
    
    # Sprawdź, czy katalog źródłowy istnieje
    if not os.path.exists(source_dir):
        log_error(f"Katalog źródłowy nie istnieje: {source_dir}", log_file)
        return False
    
    # Znajdź podkatalogi dni
    day_dirs = []
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            day_dirs.append(item_path)
    
    if not day_dirs:
        log_error("Nie znaleziono katalogów z dniami ciąży", log_file)
        return False
    
    log_info(f"Znaleziono {len(day_dirs)} katalogów dni ciąży", log_file)
    
    # Przetwórz każdy katalog dnia
    for day_dir in day_dirs:
        day_name = os.path.basename(day_dir)
        log_info(f"Przetwarzanie dnia ciąży: {day_name}", log_file)
        
        # Utwórz katalogi docelowe
        os.makedirs(os.path.join(dest_dir, "Training", day_name), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "Test", day_name), exist_ok=True)
        
        # Znajdź obrazy
        images = glob.glob(os.path.join(day_dir, "*.jpg")) + \
                glob.glob(os.path.join(day_dir, "*.jpeg")) + \
                glob.glob(os.path.join(day_dir, "*.png"))
        
        if not images:
            log_warning(f"Nie znaleziono obrazów w katalogu {day_name}", log_file)
            continue
        
        # Podziel dane na treningowe i testowe
        random.shuffle(images)
        split_idx = int(len(images) * (1 - test_split))
        
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        log_info(f"  Dane dla dnia {day_name}: {len(train_images)} treningowe, {len(test_images)} testowe", log_file)
        
        # Kopiuj pliki
        copy_images(train_images, os.path.join(dest_dir, "Training", day_name), log_file)
        copy_images(test_images, os.path.join(dest_dir, "Test", day_name), log_file)
        
        # Augmentacja (tylko dla zbioru treningowego)
        if apply_augmentation:
            augment_images(
                os.path.join(dest_dir, "Training", day_name),
                num_augmentations=max(1, int(500 / len(train_images))),
                log_file=log_file
            )
    
    log_success("Przetwarzanie danych dni ciąży zakończone", log_file)
    return True

def copy_images(image_paths, dest_dir, log_file=None):
    """
    Kopiuje pliki obrazów do wskazanego katalogu docelowego.
    Funkcja tworzy katalog docelowy jeśli nie istnieje, następnie kopiuje 
    wszystkie pliki obrazów z podanych ścieżek zachowując oryginalne nazwy 
    plików. W przypadku błędów kopiowania zapisuje informacje do dziennika zdarzeń.
    Parametry:
       ścieżki_obrazów (list): Lista ścieżek do plików obrazów do skopiowania
       katalog_docelowy (str): Ścieżka do katalogu, gdzie mają być skopiowane obrazy
       plik_logów (str, opcjonalny): Ścieżka do pliku dziennika zdarzeń
    Uwagi:
       Funkcja automatycznie tworzy katalog docelowy jeśli nie istnieje.
       Błędy kopiowania nie przerywają działania funkcji - przetwarzanie kontynuowane 
       jest dla pozostałych plików.
   """
    os.makedirs(dest_dir, exist_ok=True)
    
    for img_path in image_paths:
        try:
            filename = os.path.basename(img_path)
            dest_path = os.path.join(dest_dir, filename)
            shutil.copy2(img_path, dest_path)
        except Exception as e:
            log_error(f"Błąd podczas kopiowania {img_path}: {e}", log_file)

def augment_images(image_dir, num_augmentations=5, log_file=None):
    """
    Wykonuje wzbogacanie danych obrazowych poprzez tworzenie zmodyfikowanych kopii.
    Funkcja wczytuje wszystkie obrazy z wskazanego katalogu, następnie dla każdego 
    obrazu tworzy określoną liczbę zmodyfikowanych wersji poprzez zastosowanie 
    losowych przekształceń. Wzbogacone obrazy są zapisywane w tym samym katalogu 
    z rozszerzeniem nazwy o sufiks augmentacyjny.
    Parametry:
       katalog_obrazów (str): Ścieżka do katalogu zawierającego obrazy do wzbogacenia
       liczba_wzbogaceń (int): Liczba zmodyfikowanych kopii na jeden oryginalny obraz (domyślnie 5)
       plik_logów (str, opcjonalny): Ścieżka do pliku dziennika zdarzeń
    Uwagi:
       Funkcja przetwarza pliki w formatach JPG, JPEG i PNG. Wzbogacone obrazy 
       otrzymują nazwy z sufiksem "_aug" oraz numerem sekwencyjnym. Błędy 
       przetwarzania pojedynczych plików nie przerywają działania funkcji.
    """
    log_info(f"Augmentacja obrazów w {image_dir} ({num_augmentations} kopii na obraz)", log_file)
    
    # Znajdź obrazy
    images = glob.glob(os.path.join(image_dir, "*.jpg")) + \
            glob.glob(os.path.join(image_dir, "*.jpeg")) + \
            glob.glob(os.path.join(image_dir, "*.png"))
    
    if not images:
        log_error(f"Nie znaleziono obrazów do augmentacji w {image_dir}", log_file)
        return
    
    augmented_count = 0
    
    for img_path in images:
        try:
            # Wczytaj obraz
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            ext = os.path.splitext(img_path)[1]
            
            # Augmentacje
            for i in range(num_augmentations):
                # Losowe przekształcenia
                augmented = apply_random_augmentation(img)
                
                # Zapisz przekształcony obraz
                aug_name = f"{base_name}_aug{i+1}{ext}"
                aug_path = os.path.join(image_dir, aug_name)
                cv2.imwrite(aug_path, augmented)
                
                augmented_count += 1
        
        except Exception as e:
            log_error(f"Błąd podczas augmentacji {img_path}: {e}", log_file)
    
    log_info(f"Wygenerowano {augmented_count} augmentowanych obrazów", log_file)

def apply_random_augmentation(image):
    """
    Stosuje losowe przekształcenia geometryczne i fotometryczne do obrazu.
    Funkcja wykonuje zestaw losowych modyfikacji obrazu w celu utworzenia 
    zróżnicowanej kopii oryginalnego materiału. Każde przekształcenie jest 
    stosowane z prawdopodobieństwem 50%. Obejmuje: obrót w zakresie ±20°, 
    przesunięcie do ±30 pikseli, skalowanie w zakresie 0.8-1.2, odbicie 
    lustrzane poziome oraz modyfikację jasności i kontrastu.
    Parametry:
       obraz (numpy.ndarray): Macierz obrazu wejściowego w formacie OpenCV
    Zwraca:
       numpy.ndarray: Przekształcony obraz zachowujący wymiary oryginału
    Uwagi:
       Funkcja zachowuje oryginalne wymiary obrazu poprzez odpowiednie 
       przycinanie lub uzupełnianie po operacjach skalowania. Wszystkie 
       przekształcenia są stosowane na kopii oryginalnego obrazu.
    """
    # Kopia obrazu
    augmented = image.copy()
    
    # Losowe przekształcenia
    # 1. Obrót
    if random.random() > 0.5:
        angle = random.uniform(-20, 20)
        rows, cols = augmented.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        augmented = cv2.warpAffine(augmented, M, (cols, rows))
    
    # 2. Przesunięcie
    if random.random() > 0.5:
        tx = random.uniform(-30, 30)
        ty = random.uniform(-30, 30)
        rows, cols = augmented.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        augmented = cv2.warpAffine(augmented, M, (cols, rows))
    
    # 3. Skalowanie
    if random.random() > 0.5:
        scale = random.uniform(0.8, 1.2)
        augmented = cv2.resize(augmented, None, fx=scale, fy=scale)
        
        # Przytnij lub uzupełnij, aby zachować oryginalny rozmiar
        rows, cols = augmented.shape[:2]
        orig_rows, orig_cols = image.shape[:2]
        
        if rows > orig_rows or cols > orig_cols:
            # Przytnij
            start_row = (rows - orig_rows) // 2 if rows > orig_rows else 0
            start_col = (cols - orig_cols) // 2 if cols > orig_cols else 0
            augmented = augmented[start_row:start_row+orig_rows, start_col:start_col+orig_cols]
        else:
            # Uzupełnij
            new_img = np.zeros_like(image)
            start_row = (orig_rows - rows) // 2
            start_col = (orig_cols - cols) // 2
            new_img[start_row:start_row+rows, start_col:start_col+cols] = augmented
            augmented = new_img
    
    # 4. Odbicie lustrzane
    if random.random() > 0.5:
        augmented = cv2.flip(augmented, 1)  # Odbicie poziome
    
    # 5. Zmiana jasności/kontrastu
    if random.random() > 0.5:
        alpha = random.uniform(0.8, 1.2)  # Kontrast
        beta = random.uniform(-20, 20)    # Jasność
        augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)
    
    return augmented

def main():
    """
    Główna funkcja sterująca procesem przygotowania zestawu danych obrazowych USG klaczy.
    Funkcja koordynuje cały przepływ przetwarzania danych: analizuje argumenty 
    wiersza poleceń, konfiguruje system rejestrowania zdarzeń, tworzy strukturę 
    katalogów docelowych oraz uruchamia odpowiedni tryb przetwarzania danych 
    (klasyfikacja dwuwartościowa lub wieloklasowa dni ciąży) z opcjonalnym 
    wzbogacaniem danych.
    Zwraca:
       int: Kod zakończenia programu (0 - powodzenie, 1 - błąd)
    Uwagi:
       Funkcja automatycznie wybiera algorytm przetwarzania na podstawie 
       parametru struktury danych. Wszystkie operacje są rejestrowane 
       w pliku dziennika zdarzeń. Program kończy działanie z odpowiednim 
       kodem wyjścia informującym o powodzeniu lub niepowodzeniu operacji.
    """
    # Parsuj argumenty
    args = parse_arguments()
    
    # Skonfiguruj logowanie
    log_file = setup_logging()
    
    log_section("Przygotowanie zestawu danych USG klaczy", log_file)
    log_info(f"Katalog źródłowy: {args.source}", log_file)
    log_info(f"Katalog docelowy: {args.dest}", log_file)
    log_info(f"Podział danych testowych: {args.split}", log_file)
    log_info(f"Struktura danych: {args.structure}", log_file)
    log_info(f"Augmentacja: {'włączona' if args.augment else 'wyłączona'}", log_file)
    
    # Utwórz strukturę katalogów
    create_directory_structure(args.dest, args.structure)
    
    # Przetwarzanie danych
    if args.structure == "binary":
        success = process_binary_dataset(args.source, args.dest, args.split, args.augment, log_file)
    else:  # days
        success = process_day_dataset(args.source, args.dest, args.split, args.augment, log_file)
    
    if success:
        log_success("Przygotowanie zestawu danych zakończone pomyślnie", log_file)
        return 0
    else:
        log_error("Wystąpiły błędy podczas przygotowania zestawu danych", log_file)
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
