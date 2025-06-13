# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import os
import argparse
import json
import datetime
from config import REPORTS_DIR
from logging_utils import log_info, log_error, log_section, log_success, setup_logging
from prediction import batch_process_images, load_pregnancy_model, load_day_estimation_model
from report_generator import create_batch_report

def parse_arguments():
    """Parsuje argumenty wiersza poleceń"""
    parser = argparse.ArgumentParser(description="Przetwarzanie wsadowe obrazów USG klaczy")
    parser.add_argument("--input", "-i", required=True, help="Katalog z obrazami USG do analizy")
    parser.add_argument("--output", "-o", help="Katalog wynikowy (domyślnie utworzony w katalogu raportów)")
    parser.add_argument("--model", "-m", help="Ścieżka do modelu wykrywania ciąży")
    parser.add_argument("--report", "-r", action="store_true", help="Generuj raport zbiorczy PDF")
    parser.add_argument("--force", "-f", action="store_true", help="Nadpisz istniejące wyniki")
    
    return parser.parse_args()

def main():
    """Główna funkcja przetwarzania wsadowego"""
    # Parsuj argumenty
    args = parse_arguments()
    
    # Skonfiguruj logowanie
    log_file = setup_logging()
    
    log_section("Przetwarzanie wsadowe obrazów USG klaczy", log_file)
    log_info(f"Katalog wejściowy: {args.input}", log_file)
    
    # Sprawdź, czy katalog istnieje
    if not os.path.exists(args.input):
        log_error(f"Katalog wejściowy nie istnieje: {args.input}", log_file=log_file)
        print(f"BŁĄD: Katalog wejściowy nie istnieje: {args.input}")
        return 1
    
    # Przygotuj katalog wyjściowy
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(REPORTS_DIR, f"batch_analysis_{timestamp}")
    
    log_info(f"Katalog wyjściowy: {output_dir}", log_file)
    
    # Sprawdź, czy katalog wyjściowy istnieje i czy ma być nadpisany
    if os.path.exists(output_dir) and not args.force:
        log_error(f"Katalog wyjściowy już istnieje: {output_dir}. Użyj --force, aby nadpisać.", log_file=log_file)
        print(f"BŁĄD: Katalog wyjściowy już istnieje. Użyj --force, aby nadpisać.")
        return 1
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Ładuj modele
        pregnancy_model = load_pregnancy_model(args.model, log_file=log_file)
        
        if not pregnancy_model:
            log_error("Nie udało się załadować modelu wykrywania ciąży", log_file=log_file)
            return 1
        
        # Przetwarzaj obrazy
        log_info("Rozpoczynanie przetwarzania wsadowego...", log_file)
        results = batch_process_images(args.input, output_dir, log_file=log_file)
        
        if not results:
            log_error("Nie udało się przetworzyć obrazów", log_file=log_file)
            return 1
        
        log_success(f"Przetwarzanie zakończone. Przeanalizowano {len(results)} obrazów.", log_file)
        
        # Generuj raport zbiorczy, jeśli wymagany
        if args.report:
            log_info("Generowanie raportu zbiorczego...", log_file)
            report_path = create_batch_report(results, output_dir, log_file=log_file)
            
            if report_path:
                log_success(f"Raport zbiorczy wygenerowany: {report_path}", log_file)
            else:
                log_error("Nie udało się wygenerować raportu zbiorczego", log_file=log_file)
        
        return 0
    
    except Exception as e:
        log_error(f"Błąd podczas przetwarzania wsadowego: {e}", log_file=log_file)
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
