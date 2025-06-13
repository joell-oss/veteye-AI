# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import queue
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt  # Dodany brakujący jawny import matplotlib.pyplot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from logging_utils import log_info, log_error


def center_window(window, width=None, height=None):
    """Wyśrodkowuje okno na ekranie"""
    if width is None:
        width = window.winfo_width()
    if height is None:
        height = window.winfo_height()
    
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    
    window.geometry(f"{width}x{height}+{x}+{y}")

def create_bold_label(parent, text, **kwargs):
    """Tworzy etykietę z pogrubionym tekstem"""
    label = ttk.Label(parent, text=text, font=("TkDefaultFont", 10, "bold"), **kwargs)
    return label

def create_scrollable_frame(parent):
    """Tworzy przewijalną ramkę"""
    # Tworzenie kontenera z paskiem przewijania
    container = ttk.Frame(parent)
    
    # Dodanie paska przewijania
    scrollbar = ttk.Scrollbar(container, orient="vertical")
    scrollbar.pack(side="right", fill="y")
    
    # Tworzenie kanwy do przewijania
    canvas = tk.Canvas(container, yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    
    # Konfiguracja paska przewijania
    scrollbar.config(command=canvas.yview)
    
    # Tworzenie ramki wewnątrz kanwy
    scrollable_frame = ttk.Frame(canvas)
    
    # Dodawanie ramki do okna w kanwie
    canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    
    # Aktualizacja rozmiaru kanwy gdy zmieni się rozmiar ramki
    def configure_scroll_region(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    
    scrollable_frame.bind("<Configure>", configure_scroll_region)
    
    # Dostosowanie szerokości ramki do kanwy
    def configure_canvas_window(event):
        canvas.itemconfig(canvas_window, width=event.width)
    
    canvas.bind("<Configure>", configure_canvas_window)
    
    # Obsługa przewijania myszką (poprawiona dla większej kompatybilności)
    def on_mousewheel(event):
        # Obsługa różnych systemów operacyjnych
        if hasattr(event, 'delta'):  # Windows
            delta = event.delta
            scroll_amount = int(-1 * (delta / 120))
        elif hasattr(event, 'num'):  # Linux
            if event.num == 4:
                scroll_amount = -1
            elif event.num == 5:
                scroll_amount = 1
            else:
                scroll_amount = 0
        else:  # Inne platformy
            scroll_amount = 0
        
        canvas.yview_scroll(scroll_amount, "units")
    
    # Bindowanie zdarzeń przewijania dla różnych platform
    canvas.bind_all("<MouseWheel>", on_mousewheel)  # Windows
    canvas.bind_all("<Button-4>", on_mousewheel)    # Linux - scroll up
    canvas.bind_all("<Button-5>", on_mousewheel)    # Linux - scroll down
    
    return container, scrollable_frame

def load_and_resize_image(image_path, max_width=400, max_height=300):
    """Ładuje i zmienia rozmiar obrazu do wyświetlenia w GUI"""
    try:
        img = Image.open(image_path)
        
        # Oblicz współczynnik skalowania
        width, height = img.size
        
        # Określenie współczynnika skalowania
        if width > max_width or height > max_height:
            scale_w = max_width / width
            scale_h = max_height / height
            scale = min(scale_w, scale_h)
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Konwersja do formatu Tkinter
        img_tk = ImageTk.PhotoImage(img)
        
        return img_tk
    
    except Exception as e:
        log_error(f"Błąd podczas ładowania obrazu: {e}")
        return None

def create_progress_dialog(parent, title="Postęp", message="Proszę czekać..."):
    """Tworzy okno dialogowe z paskiem postępu"""
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.grab_set()
    
    # Wyśrodkowanie okna
    dialog_width = 300
    dialog_height = 100
    center_window(dialog, dialog_width, dialog_height)
    
    # Zapobiegaj zmianie rozmiaru i zamknięciu
    dialog.resizable(False, False)
    dialog.protocol("WM_DELETE_WINDOW", lambda: None)
    
    # Etykieta z komunikatem
    ttk.Label(dialog, text=message, wraplength=280).pack(pady=(10, 5))
    
    # Pasek postępu
    progress = ttk.Progressbar(dialog, orient="horizontal", length=250, mode="indeterminate")
    progress.pack(pady=10)
    progress.start()
    
    return dialog, progress

def run_with_progress(parent, func, args=(), kwargs={}, title="Przetwarzanie", message="Proszę czekać..."):
    """Uruchamia funkcję w osobnym wątku z oknem postępu"""
    dialog, progress = create_progress_dialog(parent, title, message)
    
    # Kolejka do przekazania wyniku z wątku
    result_queue = queue.Queue()
    
    # Funkcja wykonywana w osobnym wątku
    def threaded_func():
        try:
            result = func(*args, **kwargs)
            result_queue.put(("success", result))
        except Exception as e:
            log_error(f"Błąd w wątku: {e}")
            result_queue.put(("error", str(e)))
    
    # Funkcja sprawdzająca kolejkę
    def check_queue():
        try:
            if not result_queue.empty():
                status, result = result_queue.get()
                dialog.destroy()
                dialog.result = result  # Zapisz wynik w obiekcie dialogu
                return result
            
            # Kontynuuj sprawdzanie
            parent.after(100, check_queue)
        except Exception as e:
            log_error(f"Błąd podczas sprawdzania kolejki: {e}")
            dialog.destroy()
            messagebox.showerror("Błąd", f"Wystąpił błąd: {e}")
    
    # Uruchom funkcję w osobnym wątku
    thread = threading.Thread(target=threaded_func)
    thread.daemon = True
    thread.start()
    
    # Rozpocznij sprawdzanie kolejki
    parent.after(100, check_queue)
    
    # Zwróć dialog i postęp, aby można było je kontrolować
    return dialog, progress

def plot_figure_in_frame(frame, plot_function, figsize=(5, 4), dpi=100):
    """Tworzy wykres matplotlib w ramce Tkinter"""
    # Utwórz figurę matplotlib
    fig = plt.Figure(figsize=figsize, dpi=dpi)
    
    # Wykonaj funkcję wykresu
    plot_function(fig)
    
    # Utwórz kanwę
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    
    # Zwróć figurę i kanwę
    return fig, canvas

def show_image_with_overlay(image_path, predicted_day=None, is_pregnant=None, parent=None):
    """Wyświetla obraz z nałożonymi wynikami analizy"""
    try:
        # Utwórz nowe okno
        if parent:
            window = tk.Toplevel(parent)
        else:
            window = tk.Tk()
        
        window.title("Wynik analizy obrazu USG")
        
        # Załaduj obraz
        img = Image.open(image_path)
        width, height = img.size
        
        # Utwórz nowy obraz z tekstem
        new_height = height + 80  # Dodatkowe miejsce na tekst
        img_with_text = Image.new('RGB', (width, new_height), color='white')
        img_with_text.paste(img, (0, 0))
        
        # Przekształć na format Tkinter
        img_tk = ImageTk.PhotoImage(img_with_text)
        
        # Panel z obrazem
        panel = ttk.Label(window, image=img_tk)
        panel.image = img_tk  # Zachowaj referencję
        panel.pack(padx=10, pady=10)
        
        # Ramka na wyniki
        results_frame = ttk.Frame(window)
        results_frame.pack(fill="x", padx=10, pady=5)
        
        # Wyświetl wyniki
        if is_pregnant is not None:
            pregnancy_text = "CIĄŻA" if is_pregnant else "BRAK CIĄŻY"
            pregnancy_label = ttk.Label(
                results_frame, 
                text=pregnancy_text,
                font=("TkDefaultFont", 14, "bold"),
                foreground="green" if is_pregnant else "red"
            )
            pregnancy_label.pack(pady=5)
        
        if predicted_day is not None and is_pregnant:
            day_label = ttk.Label(
                results_frame,
                text=f"Szacowany dzień ciąży: {predicted_day}",
                font=("TkDefaultFont", 12)
            )
            day_label.pack(pady=5)
        
        # Przycisk zamknięcia
        close_button = ttk.Button(window, text="Zamknij", command=window.destroy)
        close_button.pack(pady=10)
        
        center_window(window, width+20, new_height+160)
        
        return window
    
    except Exception as e:
        log_error(f"Błąd podczas wyświetlania obrazu z nałożeniami: {e}")
        if parent:
            messagebox.showerror("Błąd", f"Wystąpił błąd: {e}")
        return None
    
def create_tooltip(widget, text):
    """Tworzy tooltip (podpowiedź) dla widgetu"""
    
    def enter(event):
        # Utwórz okno tooltipa
        x, y, _, _ = widget.bbox("insert")
        x += widget.winfo_rootx() + 25
        y += widget.winfo_rooty() + 25
        
        # Utwórz okno bez ramek
        tooltip = tk.Toplevel(widget)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{x}+{y}")
        
        # Utwórz etykietę z tekstem
        label = ttk.Label(tooltip, text=text, wraplength=250, background="#ffffe0", relief="solid", borderwidth=1)
        label.pack(padx=5, pady=5)
        
        # Zapisz referencję do tooltipa
        widget.tooltip = tooltip
    
    def leave(event):
        # Zniszcz tooltip jeśli istnieje
        if hasattr(widget, "tooltip"):
            widget.tooltip.destroy()
            delattr(widget, "tooltip")
    
    # Podłącz zdarzenia
    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)
