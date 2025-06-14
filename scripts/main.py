# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import os
import argparse
import sys
import tensorflow as tf
from logging_utils import setup_logging, log_info, log_error, log_section, log_success

"""
Konfiguracja i inicjalizacja środowiska GPU dla obliczeń TensorFlow.
Sekcja odpowiedzialna za wykrycie i prawidłowe skonfigurowanie dostępnych
kart graficznych do przyspieszenia obliczeń uczenia maszynowego. Ustawia
dynamiczną alokację pamięci GPU, która pozwala na stopniowe zwiększanie
zużycia pamięci w miarę potrzeb zamiast rezerwowania całej dostępnej pamięci
na starcie. 
W przypadku braku dostępnych kart graficznych system automatycznie przełącza
się na obliczenia procesorem, informując użytkownika o potencjalnym spadku
wydajności.
"""

# Konfiguracja GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Sprawdź dostępność GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Znaleziono {len(physical_devices)} urządzeń GPU. Używanie GPU do obliczeń.")
    # Włącz dynamiczną alokację pamięci
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("UWAGA: Nie znaleziono GPU. Program będzie działać na CPU, co może być znacznie wolniejsze.")

def parse_arguments():
    """
    Przetwarza i waliduje argumenty przekazane z linii poleceń dla systemu diagnostyki ciąży.
    Funkcja definiuje wszystkie dostępne tryby działania aplikacji oraz ich specyficzne
    parametry konfiguracyjne. Obsługuje cztery główne tryby: trenowanie modeli uczenia
    maszynowego, analizę pojedynczych obrazów USG, przetwarzanie wsadowe wielu plików
    oraz przygotowanie zestawów danych treningowych. Każdy tryb posiada dedykowane
    argumenty dopasowane do konkretnych potrzeb operacyjnych.
    Dostępne tryby działania:
    - Trening ("--train"): konfiguracja procesu uczenia modeli z opcjami wznawiania
    - Analiza ("--analyze"): diagnostyka pojedynczych obrazów ultrasonograficznych
    - Wsadowe ("--batch"): masowe przetwarzanie katalogów z obrazami USG
    - Przygotowanie ("--prepare"): organizacja i wstępne przetwarzanie danych treningowych
    Zwraca:
       Obiekt zawierający sparsowane argumenty gotowe do wykorzystania w aplikacji
    """
    parser = argparse.ArgumentParser(description="System wykrywania ciąży u klaczy")
    
    # Główne tryby działania
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--train", action="store_true", help="Tryb treningu modelu")
    modes.add_argument("--analyze", action="store_true", help="Tryb analizy obrazów USG")
    modes.add_argument("--batch", action="store_true", help="Tryb przetwarzania wsadowego")
    modes.add_argument("--prepare", action="store_true", help="Tryb przygotowania danych")
    
    # Argumenty dla treningu
    train_group = parser.add_argument_group("Argumenty treningu")
    train_group.add_argument("--model-type", choices=["pregnancy", "day"], default="pregnancy",
                            help="Typ trenowanego modelu: wykrywanie ciąży lub estymacja dnia")
    train_group.add_argument("--train-dir", help="Katalog z danymi treningowymi")
    train_group.add_argument("--resume", action="store_true", help="Wznów trening od ostatniego punktu kontrolnego")
    
    # Argumenty dla analizy
    analyze_group = parser.add_argument_group("Argumenty analizy")
    analyze_group.add_argument("--image", help="Ścieżka do obrazu USG do analizy")
    analyze_group.add_argument("--model", help="Ścieżka do modelu do używania")
    
    # Argumenty dla przetwarzania wsadowego
    batch_group = parser.add_argument_group("Argumenty przetwarzania wsadowego")
    batch_group.add_argument("--input-dir", help="Katalog wejściowy z obrazami USG")
    batch_group.add_argument("--output-dir", help="Katalog wyjściowy na wyniki")
    batch_group.add_argument("--report", action="store_true", help="Generuj raport zbiorczy PDF")
    
    # Argumenty dla przygotowania danych
    prepare_group = parser.add_argument_group("Argumenty przygotowania danych")
    prepare_group.add_argument("--source-dir", help="Katalog źródłowy z obrazami USG")
    prepare_group.add_argument("--dest-dir", help="Katalog docelowy na zestaw danych")
    prepare_group.add_argument("--structure", choices=["binary", "days"], default="binary",
                              help="Struktura danych: 'binary' dla ciąża/brak ciąży, 'days' dla dni ciąży")
    prepare_group.add_argument("--augment", action="store_true", help="Zastosuj augmentację danych")
    
    return parser.parse_args()

def train_pregnancy_model(args):
    """
    Inicjuje proces trenowania modelu sieci neuronowej do wykrywania ciąży u klaczy.
    Funkcja służy jako mostek między głównym interfejsem aplikacji a dedykowanym
    modułem treningu. Przekształca argumenty z parsera linii poleceń na format
    wymagany przez moduł treningowy, umożliwiając elastyczne konfigurowanie
    procesu uczenia. Obsługuje zarówno nowy trening jak i wznawianie przerwanego
    procesu od ostatniego punktu kontrolnego.
    Argumenty:
       args: Obiekt zawierający sparsowane argumenty z linii poleceń
             (katalog danych treningowych, flaga wznawiania)
    Zwraca:
       Wynik wykonania procesu treningu z modułu train_pregnancy_model
    """
    from train_pregnancy_model import main as train_main
    
    sys.argv = [sys.argv[0]]
    
    if args.train_dir:
        sys.argv.extend(["--train_dir", args.train_dir])
    
    if args.resume:
        sys.argv.append("--resume")
    
    return train_main()

def train_day_model(args):
    """
    Inicjuje proces trenowania modelu sieci neuronowej do szacowania wieku płodu.
    Funkcja uruchamia moduł odpowiedzialny za uczenie modelu
    estymacji dnia ciąży na podstawie obrazów ultrasonograficznych. Przekazuje
    skonfigurowane parametry do dedykowanego skryptu treningowego, umożliwiając
    precyzyjne określanie zaawansowania ciąży u klaczy. Obsługuje kontynuację
    treningu z wcześniej zapisanych punktów kontrolnych.
    Argumenty:
       args: Obiekt zawierający sparsowane argumenty z linii poleceń
             (katalog danych treningowych, flaga wznawiania treningu)
    Zwraca:
       Wynik wykonania procesu treningu z modułu train_day_estimation_model.
    Funkcja nie realizwoana w demonstratorze.   
    """
    from train_day_estimation_model import main as train_day_main
    
    sys.argv = [sys.argv[0]]
    
    if args.train_dir:
        sys.argv.extend(["--data_dir", args.train_dir])
    
    if args.resume:
        sys.argv.append("--resume")
    
    return train_day_main()

def run_analysis(args):
    """
    Wykonuje analizę obrazów USG w trybie pojedynczym lub uruchamia interfejs graficzny.
    Funkcja obsługuje dwa scenariusze użycia: analizę konkretnego pliku obrazu
    z automatycznym generowaniem raportu diagnostycznego lub uruchomienie
    graficznego interfejsu użytkownika dla interaktywnej pracy. W trybie analizy
    pojedynczego obrazu ładuje modele wykrywania ciąży i szacowania wieku płodu,
    przeprowadza pełną diagnostykę i tworzy szczegółowy raport w formacie PDF.
    Argumenty:
       args: Obiekt zawierający sparsowane argumenty z linii poleceń
             (ścieżka do obrazu, opcjonalna ścieżka do modelu)
    Zwraca:
       Kod zakończenia: 0 dla sukcesu, 1 dla błędu w trybie analizy obrazu,
       lub wynik funkcji interfejsu graficznego
    Proces obejmuje ładowanie modeli, analizę cech obrazu, predykcję oraz
    generowanie kompleksowego raportu diagnostycznego z wizualizacjami.
    """
    if args.image:
        # Analiza pojedynczego obrazu
        from prediction import analyze_and_predict, load_pregnancy_model, load_day_estimation_model
        from report_generator import create_pregnancy_report
        
        log_file = setup_logging()
        
        log_section("Analiza obrazu USG", log_file)
        
        try:
            # Ładowanie modeli
            pregnancy_model = load_pregnancy_model(args.model, log_file)
            day_model, day_mapping = load_day_estimation_model(log_file=log_file)
            
            # Analiza obrazu
            result, output_dir = analyze_and_predict(
                args.image,
                pregnancy_model,
                day_model,
                day_mapping,
                log_file=log_file
            )
            
            if result:
                # Generowanie raportu
                report_path = create_pregnancy_report(result, log_file=log_file)
                
                if report_path:
                    log_success(f"Raport wygenerowany: {report_path}", log_file)
                    print(f"Raport wygenerowany: {report_path}")
                
                return 0
            else:
                log_error("Nie udało się przeanalizować obrazu", log_file)
                return 1
        
        except Exception as e:
            log_error(f"Błąd podczas analizy obrazu: {e}", log_file)
            return 1
    
    else:
        # Uruchom interfejs graficzny
        from run_analysis_gui import main as gui_main
        
        # argumenty dla GUI
        sys.argv = [sys.argv[0]]
        
        if args.model:
            sys.argv.extend(["--model", args.model])
        
        # Uruchom GUI
        return gui_main()

def run_batch_processing(args):
    """
    Inicjuje masowe przetwarzanie wielu obrazów USG w trybie wsadowym.
    Funkcja umożliwia automatyczne przeanalizowanie całych katalogów zawierających
    obrazy ultrasonograficzne. Przekazuje skonfigurowane parametry do dedykowanego
    modułu przetwarzania wsadowego, który sekwencyjnie analizuje wszystkie pliki
    obrazów i generuje wyniki diagnostyczne. Opcjonalnie może utworzyć zbiorczy
    raport podsumowujący rezultaty całej sesji analitycznej.
    Argumenty:
       args: Obiekt zawierający sparsowane argumenty z linii poleceń
             (katalog wejściowy, katalog wyjściowy, model, flaga raportu)
    Zwraca:
       Kod zakończenia: 1 przy braku katalogu wejściowego,
       lub wynik funkcji modułu przetwarzania wsadowego
    Proces automatyzuje analizę dużych zbiorów obrazów, oszczędzając czas
    i zapewniając spójność diagnostyczną dla wszystkich przetwarzanych plików.
    """
    from batch_processing import main as batch_main
    
    sys.argv = [sys.argv[0]]
    
    if args.input_dir:
        sys.argv.extend(["--input", args.input_dir])
    else:
        print("Błąd: Nie podano katalogu wejściowego.")
        return 1
    
    if args.output_dir:
        sys.argv.extend(["--output", args.output_dir])
    
    if args.model:
        sys.argv.extend(["--model", args.model])
    
    if args.report:
        sys.argv.append("--report")
    
    return batch_main()

def prepare_dataset(args):
    """
    Inicjuje proces przygotowania i organizacji zestawu danych treningowych.
    Funkcja uruchamia dedykowany moduł odpowiedzialny za strukturyzację
    i wstępne przetwarzanie obrazów USG do celów treningu modeli uczenia
    maszynowego. Obsługuje różne schematy organizacji danych: binarny
    (ciąża/brak ciąży) oraz wieloklasowy (poszczególne dni ciąży).
    Opcjonalnie może zastosować techniki augmentacji danych w celu
    zwiększenia różnorodności zestawu treningowego.
    Argumenty:
       args: Obiekt zawierający sparsowane argumenty z linii poleceń
             (katalog źródłowy, katalog docelowy, struktura danych, augmentacja)
    Zwraca:
       Kod zakończenia: 1 przy braku katalogu źródłowego,
       lub wynik funkcji modułu prepare_dataset
    Proces obejmuje kategoryzację obrazów, walidację jakości danych,
    ewentualną augmentację oraz tworzenie uporządkowanej struktury katalogów.
    """
    from prepare_dataset import main as prepare_main
    
    sys.argv = [sys.argv[0]]
    
    if args.source_dir:
        sys.argv.extend(["--source", args.source_dir])
    else:
        print("Błąd: Nie podano katalogu źródłowego.")
        return 1
    
    if args.dest_dir:
        sys.argv.extend(["--dest", args.dest_dir])
    
    if args.structure:
        sys.argv.extend(["--structure", args.structure])
    
    if args.augment:
        sys.argv.append("--augment")
    
    return prepare_main()

def main():
    """
    Główny punkt wejścia aplikacji systemu diagnostyki ciąży u klaczy.
    Funkcja koordynuje działanie całego systemu, analizując argumenty
    przekazane z linii poleceń i delegując wykonanie do odpowiednich
    modułów specjalistycznych. Obsługuje cztery główne tryby pracy:
    trenowanie modeli uczenia maszynowego, analizę diagnostyczną obrazów,
    przetwarzanie wsadowe oraz przygotowanie zestawów danych.
    Zwraca:
       Kod zakończenia programu (0 dla sukcesu, inne wartości dla błędów)
    """
    args = parse_arguments()
    
    if args.train:
        if args.model_type == "pregnancy":
            return train_pregnancy_model(args)
        else:  # day
            return train_day_model(args)
    
    elif args.analyze:
        return run_analysis(args)
    
    elif args.batch:
        return run_batch_processing(args)
    
    elif args.prepare:
        return prepare_dataset(args)
    
    return 0

if __name__ == "__main__":
    """
    Punkt uruchomieniowy skryptu - wykonuje główną funkcję i kończy program
    z odpowiednim kodem zakończenia sygnalizującym sukces lub błąd operacji.
    """
    exit_code = main()
    exit(exit_code)
