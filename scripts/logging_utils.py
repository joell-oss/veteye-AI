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
    """
    Konfiguruje system logowania z automatyczną rotacją plików dziennika.
    Funkcja tworzy i zarządza plikami dziennika z ograniczeniem rozmiaru i liczby.
    Gdy plik osiągnie maksymalny rozmiar, zostaje przeniesiony z dodaniem znacznika
    czasu, a tworzony jest nowy plik. System automatycznie usuwa najstarsze pliki
    dziennika, aby utrzymać określoną liczbę archiwów. Nazwy plików zawierają
    aktualną datę dla łatwiejszej identyfikacji.
    Argumenty:
       log_dir: Katalog docelowy dla plików dziennika
       max_log_files: Maksymalna liczba przechowywanych plików dziennika
       max_log_size_mb: Maksymalny rozmiar pojedynczego pliku w megabajtach
    Zwraca:
       Ścieżka do aktualnego pliku dziennika
    """
    
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
    """
    Zapisuje komunikat z znacznikiem czasowym do pliku dziennika i opcjonalnie na konsolę.
    Funkcja dodaje do wiadomości aktualną datę i godzinę, a następnie zapisuje
    sformatowany tekst do określonego pliku dziennika. Jeśli nie podano pliku,
    automatycznie konfiguruje system logowania. Komunikat może być jednocześnie
    wyświetlony na konsoli dla natychmiastowego podglądu.
    Argumenty:
       message: Treść komunikatu do zapisania
       log_file: Ścieżka do pliku dziennika (opcjonalne, auto-konfiguracja)
       print_to_console: Czy wyświetlić komunikat również na konsoli
    Funkcja obsługuje błędy zapisu i wyświetla informacje diagnostyczne
    w przypadku problemów z dostępem do pliku.
    """
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
    """
    Zapis wizualnie wyróżnionego nagłóweka sekcji w dzienniku zdarzeń.
    Tworzy czytelny separator z podkreśleniem dla oznaczenia początku
    nowej sekcji w pliku dziennika, ułatwiając nawigację i organizację
    zapisanych informacji.
    """
    separator = "=" * len(section_name)
    log_message(f"\n{separator}\n{section_name}\n{separator}", log_file)

def log_info(info_message, log_file=None):
    """
    Zapis komunikatu informacyjnego z prefiksem INFO w dzienniku.
    Przeznaczona do logowania standardowych informacji o przebiegu
    działania aplikacji i stanie systemu.
    """
    log_message(f"INFO: {info_message}", log_file)

def log_warning(warning_message, log_file=None):
    """
    Zapis komunikatu ostrzegawczego z prefiksem OSTRZEŻENIE w dzienniku.
    Służy do rejestrowania sytuacji problemowych, które nie przerywają
    działania, ale wymagają uwagi użytkownika lub administratora.
    """
    log_message(f"OSTRZEŻENIE: {warning_message}", log_file)

def log_success(success_message, log_file=None):
    """
    Zapis komunikatu o pomyślnym zakończeniu operacji z prefiksem SUKCES.
    Przeznaczona do oznaczania ukończonych zadań i operacji wykonanych
    bez błędów, ułatwiając śledzenie postępów w dzienniku.
    """
    log_message(f"SUKCES: {success_message}", log_file)
