# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import os
import argparse
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from analysis_gui import AnalysisGUI
from config import CHECKPOINTS_DIR
from logging_utils import setup_logging, log_info, log_error

def parse_arguments():
    """
    Analizuje i przetwarza argumenty przekazane z wiersza poleceń.
    Funkcja konfiguruje parser argumentów dla interfejsu graficznego systemu 
    analizy obrazów USG klaczy. Umożliwia użytkownikowi opcjonalne wskazanie 
    ścieżki do modelu wykrywania ciąży poprzez parametr --model lub -m.
    Zwraca:
       argparse.Namespace: Obiekt zawierający sparsowane argumenty wiersza poleceń
    Uwagi:
       Parser automatycznie generuje pomoc dostępną poprzez parametr --help.
       Wszystkie argumenty są opcjonalne - aplikacja może działać bez dodatkowych 
       parametrów używając domyślnych ustawień.
    """
    parser = argparse.ArgumentParser(description="Interfejs graficzny do analizy USG ciąży klaczy")
    parser.add_argument("--model", "-m", help="Ścieżka do modelu wykrywania ciąży")
    
    return parser.parse_args()

def find_latest_model():
    """
    Wyszukuje najnowszy dostępny model uczenia maszynowego w katalogu punktów kontrolnych.
    Funkcja przeszukuje katalog z zapisanymi modelami w poszukiwaniu plików 
    w formacie Keras (.keras) zawierających w nazwie słowa "final" lub "finetuned". 
    Spośród znalezionych modeli wybiera najnowszy na podstawie daty modyfikacji pliku.
    Zwraca:
       str: Ścieżka do najnowszego modelu lub None jeśli nie znaleziono żadnego modelu
    Uwagi:
       Funkcja priorytetowo traktuje modele końcowe i dostrojone. W przypadku błędu 
       dostępu do katalogu lub jego braku zwraca None i wyświetla komunikat o błędzie. 
       Porównanie dat odbywa się na podstawie czasu ostatniej modyfikacji plików.
    """
    try:
        # Szukaj modeli końcowych
        model_files = []
        for filename in os.listdir(CHECKPOINTS_DIR):
            if filename.endswith(".keras") and ("final" in filename or "finetuned" in filename):
                model_files.append(os.path.join(CHECKPOINTS_DIR, filename))
        
        if model_files:
            # Zwróć najnowszy model
            return max(model_files, key=os.path.getmtime)
        
        return None
    
    except Exception as e:
        print(f"Błąd podczas szukania modelu: {e}")
        return None

def main():
    """
    Główna funkcja inicjalizująca i uruchamiająca graficzny interfejs użytkownika.
    Funkcja koordynuje proces uruchomienia aplikacji: analizuje argumenty wiersza 
    poleceń, konfiguruje system rejestrowania zdarzeń, wyszukuje odpowiedni model 
    uczenia maszynowego, tworzy główne okno aplikacji z ustawionym motywem graficznym 
    oraz inicjalizuje interfejs analizy obrazów USG.
    Uwagi:
       Jeśli nie wskazano ścieżki do modelu w argumentach, funkcja automatycznie 
       wyszukuje najnowszy dostępny model w katalogu punktów kontrolnych. 
       Aplikacja może działać bez modelu, jednak funkcje analizy będą niedostępne. 
       Wykorzystuje bibliotekę Tkinter z motywem 'clam' dla nowoczesnego wyglądu.
    """
    # Parsuj argumenty
    args = parse_arguments()
    
    # Skonfiguruj logowanie
    log_file = setup_logging()
    
    # Znajdź model do użycia
    model_path = args.model
    
    if not model_path:
        # Szukaj najnowszego modelu
        model_path = find_latest_model()
        
        if model_path:
            log_info(f"Używanie najnowszego modelu: {model_path}", log_file)
        else:
            log_error("Nie znaleziono modelu. Można kontynuować, ale analiza nie będzie dostępna.", log_file)
    
    # Utwórz główne okno
    root = tk.Tk()
    root.title("System wykrywania ciąży u klaczy")
    
    # Utwórz styl
    style = ttk.Style()
    style.theme_use('clam')  # opcje: 'alt', 'default', 'classic'
    
    # Utwórz i uruchom GUI
    app = AnalysisGUI(root, model_path)
    
    # Uruchom pętlę zdarzeń
    root.mainloop()

if __name__ == "__main__":
    main()
