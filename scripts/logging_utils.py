# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import os
import datetime
import traceback
from config import LOGS_DIR

def setup_logging(log_dir=LOGS_DIR, max_log_files=10, max_log_size_mb=10):
    """Konfiguruje rotacyjny system logowania"""
    
    # Utwórz katalog na logi, jeśli nie istnieje
    os.makedirs(log_dir, exist_ok=True)
    
    # Utwórz nazwę pliku z datą
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"training_log_{current_date}.txt")
    
    # Funkcja sprawdzająca i rotująca logi
    def rotate_logs():
        if os.path.exists(log_file) and os.path.getsize(log_file) > max_log_size_mb * 1024 * 1024:
            # Przesuń istniejący plik z dodaniem czasu
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            rotated_file = log_file.replace(".txt", f"_{timestamp}.txt")
            os.rename(log_file, rotated_file)
            
            # Usuń najstarsze pliki, jeśli jest ich za dużo
            all_logs = sorted([f for f in os.listdir(log_dir) if f.startswith("training_log_")])
            if len(all_logs) > max_log_files:
                for old_log in all_logs[:-max_log_files]:
                    try:
                        os.remove(os.path.join(log_dir, old_log))
                    except Exception as e:
                        print(f"Nie można usunąć starego logu {old_log}: {e}")
    
    # Rotacja na starcie
    rotate_logs()
    
    return log_file

def log_message(message, log_file=None, print_to_console=True):
    """Zapisuje wiadomość do pliku logu z datą i godziną"""
    if log_file is None:
        log_file = setup_logging()
        
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    
    if print_to_console:
        print(formatted_message)
    
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(formatted_message + "\n")
    except Exception as e:
        print(f"Błąd podczas zapisywania do logu: {e}")
        traceback.print_exc()

def log_error(error_message, exception=None, log_file=None):
    """Loguje błąd wraz ze śladem wywołań, jeśli podano wyjątek"""
    error_log = f"BŁĄD: {error_message}"
    
    if exception:
        error_log += f"\n{str(exception)}\n"
        error_log += traceback.format_exc()
    
    log_message(error_log, log_file)

def log_section(section_name, log_file=None):
    """Loguje nagłówek sekcji z podkreśleniem dla lepszej czytelności"""
    separator = "=" * len(section_name)
    log_message(f"\n{separator}\n{section_name}\n{separator}", log_file)

def log_info(info_message, log_file=None):
    """Loguje informację"""
    log_message(f"INFO: {info_message}", log_file)

def log_warning(warning_message, log_file=None):
    """Loguje ostrzeżenie"""
    log_message(f"OSTRZEŻENIE: {warning_message}", log_file)

def log_success(success_message, log_file=None):
    """Loguje sukces"""
    log_message(f"SUKCES: {success_message}", log_file)
