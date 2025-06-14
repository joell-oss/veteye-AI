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
from screeninfo import get_monitors


def center_window(window, width=None, height=None, monitor_index=1):
    """
    Funkcja do wyśrodkowania okna aplikacji na wybranym monitorze w systemie wielomonitorowym.
    Funkcjonalność:
    - Automatycznie wykrywa wszystkie dostępne monitory w systemie
    - Pozycjonuje okno na środku wskazanego monitora (domyślnie drugi monitor)
    - Pobiera wymiary okna lub używa przekazanych parametrów szerokości/wysokości
    - Oblicza współrzędne środka ekranu uwzględniając pozycję monitora
    - Zabezpiecza przed błędami gdy wybrany monitor nie istnieje (przełącza na pierwszy)
    - Ustawia geometrię okna z precyzyjnymi współrzędnymi pozycji
    Obliczenia pozycji:
    - x = pozycja_monitora_x + (szerokość_monitora - szerokość_okna) / 2
    - y = pozycja_monitora_y + (wysokość_monitora - wysokość_okna) / 2
    Parametry:
    - window: obiekt okna Tkinter do wyśrodkowania
    - width: szerokość okna (opcjonalna, pobierana automatycznie)
    - height: wysokość okna (opcjonalna, pobierana automatycznie) 
    - monitor_index: indeks monitora docelowego (domyślnie 1 = drugi monitor)
    Zastosowanie: pozycjonowanie okien aplikacji medycznych na dedykowanym monitorze
    w środowisku wieloekranowym (np. monitor główny + monitor dla obrazów USG)
    """
    if width is None:
        width = window.winfo_width()
    if height is None:
        height = window.winfo_height()

    monitors = get_monitors()

    # Sprawdź czy dany monitor istnieje
    if monitor_index >= len(monitors):
        monitor_index = 0  # domyślnie pierwszy monitor

    monitor = monitors[monitor_index]
    screen_x, screen_y = monitor.x, monitor.y
    screen_width, screen_height = monitor.width, monitor.height

    x = screen_x + (screen_width - width) // 2
    y = screen_y + (screen_height - height) // 2

    window.geometry(f"{width}x{height}+{x}+{y}")


def create_bold_label(parent, text, **kwargs):
    """
    Funkcja pomocnicza do tworzenia etykiet z pogrubionym tekstem w interfejsie Tkinter.
    Funkcjonalność:
    - Tworzy widget ttk.Label z automatycznie ustawioną pogrubioną czcionką
    - Wykorzystuje domyślną czcionkę systemową TkDefaultFont w rozmiarze 10pt
    - Przyjmuje dowolne dodatkowe parametry formatowania przez **kwargs
    - Zwraca gotowy obiekt etykiety do umieszczenia w kontenerze
    Parametry:
    - parent: widget nadrzędny (ramka, okno) do umieszczenia etykiety
    - text: treść tekstowa do wyświetlenia
    - **kwargs: dodatkowe opcje formatowania (kolor, padding, justowanie itp.)
    Zastosowanie: 
    Upraszcza tworzenie nagłówków, tytułów sekcji i ważnych etykiet 
    w interfejsie użytkownika bez konieczności wielokrotnego definiowania 
    parametrów czcionki. Szczególnie przydatne dla etykiet opisujących 
    pola formularzy w aplikacjach medycznych.
    Przykład użycia:
    title_label = create_bold_label(frame, "Dane badanego:", foreground="blue")
    """
    label = ttk.Label(parent, text=text, font=("TkDefaultFont", 10, "bold"), **kwargs)
    return label

def create_scrollable_frame(parent):
    """
    Funkcja tworzy przewijalną ramkę z obsługą kółka myszy dla interfejsu Tkinter.
    Architektura komponentów:
    - Kontener główny zawierający pasek przewijania i kanwę
    - Pionowy pasek przewijania umieszczony po prawej stronie
    - Kanwa (Canvas) służąca jako obszar przewijania
    - Wewnętrzna ramka (scrollable_frame) do umieszczania widgetów
    Mechanizm przewijania:
    - Automatyczna aktualizacja obszaru przewijania przy zmianie zawartości
    - Dynamiczne dopasowanie szerokości ramki do kanwy
    - Obsługa przewijania kółkiem myszy dla różnych systemów operacyjnych
    - Kompatybilność z Windows (MouseWheel), Linux (Button-4/5) i innymi platformami
    Funkcje pomocnicze:
    - configure_scroll_region(): aktualizuje zakres przewijania
    - configure_canvas_window(): dopasowuje szerokość do kontenera
    - on_mousewheel(): obsługuje przewijanie kółkiem myszy wieloplatformowo
    Zwracane wartości:
    - container: główny kontener do umieszczenia w oknie
    - scrollable_frame: ramka do dodawania widgetów potomnych
    Zastosowanie: 
    Tworzenie przewijalnych formularzy, list wyników diagnostycznych, 
    długich raportów medycznych w aplikacjach z interfejsem graficznym.
    Szczególnie przydatne gdy zawartość przekracza dostępną przestrzeń ekranu.
    """
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
    """
    Funkcja ładuje obraz z dysku i skaluje go do wyświetlenia w interfejsie graficznym.
    Proces przetwarzania obrazu:
    - Otwiera plik obrazu z podanej ścieżki za pomocą PIL/Pillow
    - Sprawdza czy wymiary przekraczają maksymalne wartości (400x300 px domyślnie)
    - Oblicza proporcjonalny współczynnik skalowania zachowując proporcje
    - Używa algorytmu LANCZOS dla wysokiej jakości przeskalowania
    - Konwertuje finalny obraz do formatu ImageTk.PhotoImage dla Tkinter
    Algorytm skalowania:
    - Porównuje współczynniki skalowania dla szerokości i wysokości
    - Wybiera mniejszy współczynnik aby obraz zmieścił się w zadanych granicach
    - Zachowuje oryginalny stosunek proporcji obrazu (brak deformacji)
    - Pomija skalowanie jeśli obraz jest już wystarczająco mały
    Parametry:
    - image_path: ścieżka do pliku obrazu na dysku
    - max_width: maksymalna szerokość w pikselach (domyślnie 400)
    - max_height: maksymalna wysokość w pikselach (domyślnie 300)
    Zwraca: obiekt ImageTk.PhotoImage gotowy do wyświetlenia lub None przy błędzie
    Zastosowanie: 
    Przygotowanie obrazów USG do podglądu w aplikacji diagnostycznej 
    bez przekraczania rozmiarów interfejsu użytkownika.
    """
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
    """
    Funkcja tworzy modalne okno dialogowe z animowanym paskiem postępu dla długotrwałych operacji.
    Właściwości okna dialogowego:
    - Okno podrzędne (Toplevel) przypisane do okna nadrzędnego
    - Modalność - blokuje interakcję z głównym oknem (grab_set)
    - Automatyczne wyśrodkowanie na wybranym monitorze (domyślnie drugim)
    - Zablokowana możliwość zmiany rozmiaru i zamknięcia przez użytkownika
    - Wymiary stałe: 300x100 pikseli
    Komponenty interfejsu:
    - Etykieta z komunikatem o postępie (z zawijaniem tekstu do 280px)
    - Pasek postępu w trybie nieokreślonym (indeterminate) - animacja ciągła
    - Automatyczne uruchomienie animacji paska postępu
    Parametry konfiguracyjne:
    - parent: okno nadrzędne dla dialogu
    - title: tytuł okna (domyślnie "Postęp")
    - message: tekst komunikatu (domyślnie "Proszę czekać...")
    Zwracane obiekty:
    - dialog: referencja do okna dialogowego
    - progress: referencja do paska postępu
    Zastosowanie:
    Informowanie użytkownika o trwających operacjach jak analiza obrazów USG,
    generowanie raportów PDF, trenowanie modeli AI. Zapobiega przypadkowemu
    zamknięciu aplikacji podczas krytycznych procesów.
    """
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.grab_set()
    
    # Wyśrodkowanie okna
    dialog_width = 300
    dialog_height = 100
    center_window(dialog, dialog_width, dialog_height, monitor_index=1)
    
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
    """
    Funkcja wykonuje długotrwałe operacje w tle z wyświetlaniem okna postępu bez blokowania interfejsu.
    Architektura wielowątkowa:
    - Główny wątek GUI pozostaje responsywny dla użytkownika
    - Osobny wątek (daemon) wykonuje zadaną funkcję w tle
    - Kolejka (Queue) służy do bezpiecznej komunikacji między wątkami
    - Cykliczne sprawdzanie wyników co 100ms za pomocą parent.after()
    Mechanizm działania:
    1. Tworzy modalne okno dialogowe z paskiem postępu
    2. Uruchamia zadaną funkcję w osobnym wątku z przekazanymi argumentami
    3. Monitoruje stan wykonania przez kolejkę komunikatów
    4. Automatycznie zamyka dialog po zakończeniu operacji
    5. Obsługuje błędy i przekazuje wyniki z powrotem do głównego wątku
    Obsługa wyników:
    - Sukces: zwraca wynik funkcji przez result_queue
    - Błąd: loguje błąd i wyświetla komunikat użytkownikowi
    - Wynik zapisywany w atrybucie dialog.result
    Parametry:
    - parent: okno nadrzędne GUI
    - func: funkcja do wykonania w tle
    - args: argumenty pozycyjne dla funkcji
    - kwargs: argumenty nazwane dla funkcji
    - title: tytuł okna postępu
    - message: komunikat wyświetlany użytkownikowi
    Zastosowanie: 
    Analiza obrazów USG, generowanie raportów PDF, predykcje modeli AI
    bez zamrażania interfejsu użytkownika podczas obliczeń.
    """
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
    """
    Funkcja integruje wykresy matplotlib z interfejsem Tkinter poprzez osadzenie w ramce.
    Proces tworzenia wykresu:
    - Tworzy obiekt Figure matplotlib z określonymi wymiarami i rozdzielczością
    - Wykonuje przekazaną funkcję rysującą wykres na utworzonej figurze
    - Konwertuje figurę matplotlib na widget Tkinter przy użyciu FigureCanvasTkAgg
    - Automatycznie wypełnia dostępną przestrzeń w ramce nadrzędnej
    Konfiguracja wyświetlania:
    - Domyślny rozmiar: 5x4 cale
    - Domyślna rozdzielczość: 100 DPI
    - Widget kanwy rozciąga się na całą dostępną przestrzeń (fill=BOTH, expand=True)
    Parametry:
    - frame: ramka Tkinter do umieszczenia wykresu
    - plot_function: funkcja przyjmująca obiekt Figure i rysująca wykres
    - figsize: krotka (szerokość, wysokość) w calach
    - dpi: rozdzielczość wykresu w punktach na cal
    Zwracane obiekty:
    - fig: obiekt Figure matplotlib do dalszych modyfikacji
    - canvas: kanwa Tkinter z osadzonym wykresem
    Zastosowanie:
    Wyświetlanie wykresów analizy obrazów USG, histogramów cech diagnostycznych,
    macierzy pomyłek modeli AI bezpośrednio w oknie aplikacji medycznej
    bez konieczności otwierania zewnętrznych okien matplotlib.
    """
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
    """
    Funkcja wyświetla obraz USG z nałożonymi wynikami analizy diagnostycznej w osobnym oknie.
    Proces wyświetlania:
    - Tworzy nowe okno dialogowe (Toplevel) lub główne okno aplikacji
    - Ładuje oryginalny obraz USG i rozszerza go o 80 pikseli na dole
    - Dodaje białe tło pod obrazem dla umieszczenia tekstów wyników
    - Konwertuje obraz do formatu ImageTk.PhotoImage dla Tkinter
    Prezentacja wyników:
    - Status ciąży wyświetlany dużą, pogrubioną czcionką (zielona/czerwona)
    - Szacowany dzień ciąży pokazywany tylko przy pozytywnym wyniku
    - Wyniki umieszczone w osobnej ramce pod obrazem
    - Przycisk "Zamknij" na dole okna
    Układanie elementów:
    - Obraz USG na górze z ramką 10px
    - Ramka wyników wypełniająca szerokość okna
    - Automatyczne wyśrodkowanie na wybranym monitorze
    - Rozmiar okna dopasowany do obrazu plus marginesy (20px + 160px)
    Parametry:
    - image_path: ścieżka do pliku obrazu USG
    - predicted_day: przewidywany dzień ciąży (opcjonalny)
    - is_pregnant: status ciąży True/False (opcjonalny)
    - parent: okno nadrzędne dla modalności (opcjonalny)
    Zastosowanie:
    Prezentacja końcowych wyników analizy AI w czytelnej formie graficznej
    z zachowaniem oryginalnego obrazu USG i dodanymi adnotacjami diagnostycznymi.
    """
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
        
        center_window(window, width+20, new_height+160, monitor_index=1)
        
        return window
    
    except Exception as e:
        log_error(f"Błąd podczas wyświetlania obrazu z nałożeniami: {e}")
        if parent:
            messagebox.showerror("Błąd", f"Wystąpił błąd: {e}")
        return None
    
def create_tooltip(widget, text):
    """
   Tworzy interaktywną podpowiedź (tooltip) dla dowolnego widgetu Tkinter.
   Funkcja dodaje do widgetu możliwość wyświetlania pomocniczego tekstu
   po najechaniu myszką. Tooltip pojawia się jako małe okno obok kursora
   i znika po opuszczeniu widgetu.
   Argumenty:
       widget: Widżet Tkinter, do którego ma zostać dodany tooltip
       text: Tekst podpowiedzi do wyświetlenia
   """
    
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
