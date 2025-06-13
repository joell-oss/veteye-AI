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
    """Parsuje argumenty wiersza poleceń"""
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
    """Uruchamia trening modelu wykrywania ciąży"""
    from train_pregnancy_model import main as train_main
    
    # Przygotuj argumenty do skryptu treningu
    sys.argv = [sys.argv[0]]
    
    if args.train_dir:
        sys.argv.extend(["--train_dir", args.train_dir])
    
    if args.resume:
        sys.argv.append("--resume")
    
    # Uruchom skrypt treningu
    return train_main()

def train_day_model(args):
    """Uruchamia trening modelu szacowania dnia ciąży"""
    from train_day_estimation_model import main as train_day_main
    
    # Przygotuj argumenty do skryptu treningu
    sys.argv = [sys.argv[0]]
    
    if args.train_dir:
        sys.argv.extend(["--data_dir", args.train_dir])
    
    if args.resume:
        sys.argv.append("--resume")
    
    # Uruchom skrypt treningu
    return train_day_main()

def run_analysis(args):
    """Uruchamia analizę pojedynczego obrazu lub interfejs graficzny"""
    if args.image:
        # Analiza pojedynczego obrazu
        from prediction import analyze_and_predict, load_pregnancy_model, load_day_estimation_model
        from report_generator import create_pregnancy_report
        
        # Skonfiguruj logowanie
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
        
        # Przygotuj argumenty dla GUI
        sys.argv = [sys.argv[0]]
        
        if args.model:
            sys.argv.extend(["--model", args.model])
        
        # Uruchom GUI
        return gui_main()

def run_batch_processing(args):
    """Uruchamia przetwarzanie wsadowe"""
    from batch_processing import main as batch_main
    
    # Przygotuj argumenty
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
    
    # Uruchom przetwarzanie wsadowe
    return batch_main()

def prepare_dataset(args):
    """Uruchamia przygotowanie zestawu danych"""
    from prepare_dataset import main as prepare_main
    
    # Przygotuj argumenty
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
    
    # Uruchom przygotowanie danych
    return prepare_main()

def main():
    """Główna funkcja programu"""
    # Parsuj argumenty
    args = parse_arguments()
    
    # Uruchom odpowiedni tryb
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
    exit_code = main()
    exit(exit_code)
