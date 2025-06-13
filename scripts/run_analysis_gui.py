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
    """Parsuje argumenty wiersza poleceń"""
    parser = argparse.ArgumentParser(description="Interfejs graficzny do analizy USG ciąży klaczy")
    parser.add_argument("--model", "-m", help="Ścieżka do modelu wykrywania ciąży")
    
    return parser.parse_args()

def find_latest_model():
    """Znajduje najnowszy dostępny model"""
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
    """Główna funkcja uruchamiająca interfejs graficzny"""
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
    style.theme_use('clam')  # Można zmienić na 'alt', 'default', 'classic'
    
    # Utwórz i uruchom GUI
    app = AnalysisGUI(root, model_path)
    
    # Uruchom pętlę zdarzeń
    root.mainloop()

if __name__ == "__main__":
    main()
