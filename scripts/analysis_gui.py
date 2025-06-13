# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import tkinter as tk
import os
import threading
import json
import datetime
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import tempfile
from PIL import Image, ImageTk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from config import IMAGE_SIZE, REPORTS_DIR, CHECKPOINTS_DIR
from logging_utils import log_info, log_error, log_success, setup_logging
from data_loader import load_and_preprocess_image
from prediction import predict_pregnancy, estimate_pregnancy_day, analyze_and_predict, batch_process_images
from report_generator import create_pregnancy_report, create_batch_report
from gui_utils import (
    center_window, create_bold_label, create_scrollable_frame, 
    load_and_resize_image, run_with_progress, plot_figure_in_frame,
    show_image_with_overlay, create_tooltip
)
from diagnostic_utils import generate_pdf_report, generate_description, analyze_image_features
from prediction import predict_image
from tkinter import messagebox 

import glob



class AnalysisGUI:

    
    def __init__(self, master, model_path=None):
        self.master = master
        self.master.title("veteye.AI - System predykcji ciąży klaczy na podstawie diagnostyki obrazowej USG")

        # Ustawienie stylu ttk
        style = ttk.Style()
        #style.theme_use('xpnative')  # lub 'alt', 'default', 'vista' (na Windows), 'xpnative'
        style.theme_use('alt')
        # Przykład niestandardowego stylu przycisku
        style.configure('TButton',
                        font=('Segoe UI', 10),
                        foreground='navy',
                        padding=6)

        # Styl dla ramek i etykiet
        style.configure('TLabel',
                        font=('Segoe UI', 10),
                        foreground='navy'),
        style.configure('TEntry',
                        font=('Segoe UI', 10),
                        foreground='navy'
                        )
        style.configure('TLabelframe.Label',
                        font=('Segoe UI', 10, 'bold'),
                        foreground='navy'
                        )
                        
        style.configure('TNotebook.Tab', font=('Segoe UI', 10),
                        foreground='navy')


        self.master.geometry("900x700")
        self.master.minsize(800, 600)
        self.master.state('zoomed')

        self.model_path = model_path
        self.log_file = setup_logging()
        log_info("Uruchomiono interfejs analizy obrazów USG", self.log_file)

        self.pregnancy_model = None
        self.day_model = None
        self.day_mapping = {}

        self.klacz_name = tk.StringVar()
        self.klacz_age = tk.StringVar()
        self.estimated_day = tk.StringVar()

        self.setup_gui()
        self.load_models()
        
    def get_latest_model_path():
        files = glob.glob("checkpoints/USGEquina-Pregna_v*.keras")
        if not files:
            return None
        return sorted(files)[-1]  # zakładamy, że sortowanie alfabetyczne odpowiada wersjom

        # i później:
        self.model_path = get_latest_model_path()

    def center_window(self, window, width=600, height=400):
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        window.geometry(f"{width}x{height}+{x}+{y}")

    def show_feature_legend(self):
        """Okno z wyjaśnieniem cech obrazu"""
        legend_window = tk.Toplevel(self.master)
        legend_window.title("Legenda cech obrazu")
        legend_window.geometry("600x400")
    
        # Ramka i przewijane pole tekstowe
        frame = ttk.Frame(legend_window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        text = tk.Text(frame, wrap=tk.WORD)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.config(yscrollcommand=scrollbar.set)

        # Wstaw tekst wyjaśniający – identyczny jak w PDF
        feature_explanation = [
            "• mean_intensity – średnia jasność pikseli obrazu.\n  Typowy zakres: 0.2 – 0.5. Zbyt niska = niedoświetlony, zbyt wysoka = prześwietlony.",
            "• std_intensity – odchylenie standardowe jasności.\n  Typowy zakres: 0.1 – 0.3. Za niska = mało informacji, za wysoka = szum.",
            "• contrast – kontrast do jasności.\n  Typowy zakres: > 0.4. Większy = lepsza czytelność struktur.",
            "• entropy – złożoność tekstury.\n  Typowy zakres: 2.5 – 4.5. Za niska = ubogi obraz, za wysoka >5 = szum.",
            "• edge_magnitude – siła krawędzi.\n  Typowy zakres: 0.2 – 0.5. Za niska = niewyraźne struktury, za wysoka = artefakty.",
            "• potential_fluid_ratio – jasne obszary (płyn).\n  Typowy zakres: 0.05 – 0.3. Więcej = możliwa obecność pęcherzyka.",
            "• potential_tissue_ratio – ciemne obszary (tkanki).\n  Typowy zakres: 0.4 – 0.8. Za mało = trudna ocena anatomii."
        ]
    
        text.insert(tk.END, "\n\n".join(feature_explanation))
        text.config(state='disabled')

        # Wyśrodkuj okno względem głównego
        self.center_window(legend_window, width=600, height=400)


    def setup_gui(self):
        style = ttk.Style()
        self.master.title("veteye.AI - System diagnostyki USG klaczy - demonstrator (wersja: 1.01.001)")
        self.master.geometry("1200x800")
        self.master.minsize(1000, 700)

        # === GŁÓWNY PANEL ===
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === GÓRNY PANEL (3 sekcje) ===
        top_frame = tk.Frame(main_frame)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False, padx=10, pady=10)

        # 1. Dane klaczy
        mare_data_frame = ttk.LabelFrame(top_frame, text="Dane klaczy")
        mare_data_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.klacz_name = tk.StringVar()
        self.klacz_age = tk.StringVar()
        self.estimated_day = tk.StringVar()

        ttk.Label(mare_data_frame, text="Imię:").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Entry(mare_data_frame, textvariable=self.klacz_name, width=20).pack(padx=5)
        ttk.Label(mare_data_frame, text="Wiek (lata):").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Entry(mare_data_frame, textvariable=self.klacz_age, width=20).pack(padx=5)
        ttk.Label(mare_data_frame, text="Szacowany dzień cyklu:").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Entry(mare_data_frame, textvariable=self.estimated_day, width=20).pack(padx=5)

        # 2. Wykres cech obrazu
        self.features_frame = ttk.LabelFrame(top_frame, text="Cechy obrazu")
        self.features_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # (Wypełniany później np. przez self.plot_features())

        # 3. Obraz USG
        self.image_frame = ttk.LabelFrame(top_frame, text="Obraz USG")
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.image_label = tk.Label(self.image_frame, text="Brak obrazu", bg="gray")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # === DOLNY PANEL (Opis) ===
        bottom_frame = ttk.LabelFrame(main_frame, text="Opis diagnostyczny")
        bottom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.result_text = tk.Text(bottom_frame, wrap=tk.WORD, height=15)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(bottom_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)

        # Najpierw zdefiniuj styl dla ramki przycisków
        style.configure('ButtonPanel.TFrame',
                        background='navy')  # Możesz użyć dowolnego koloru tła

        # === DOLNY PANEL (przyciski i status) ===
        button_frame = ttk.Frame(self.master, style='ButtonPanel.TFrame')
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # Możesz także utworzyć styl dla przycisków w panelu (opcjonalnie)
        style.configure('Panel.TButton',
                        font=('Segoe UI', 10),
                        foreground='navy',
                        padding=6)

        # Zastosuj styl do przycisków
        self.load_image_button = ttk.Button(button_frame, text="Wczytaj obraz", 
                                            command=self.load_image,
                                            padding=(5, 1), 
                                            style='Panel.TButton')
        self.load_image_button.pack(side=tk.LEFT, padx=5)

        self.analyze_button = ttk.Button(button_frame, text="Analizuj", 
                                        command=self.analyze_current_image, 
                                        state=tk.DISABLED,
                                        padding=(5, 1),
                                        style='Panel.TButton')
        self.analyze_button.pack(side=tk.LEFT, padx=5)

        self.legend_button = ttk.Button(button_frame, text="Legenda cech", 
                                        command=self.show_feature_legend, 
                                        state=tk.DISABLED,
                                        padding=(5, 1),
                                        style='Panel.TButton')
        self.legend_button.pack(side=tk.LEFT, padx=5)

        # === Przycisk zapisu raportu i zakończenia ===
        #button_frame = ttk.Frame(self.master)
        #button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # Funkcja zamykająca aplikację z potwierdzeniem
        def confirm_exit():
            if messagebox.askyesno("Zakończ", "Czy na pewno chcesz zakończyć pracę?"):
                self.master.quit()

        self.exit_button = ttk.Button(button_frame, padding=(5, 1), text="Zakończ", command=confirm_exit)
        self.exit_button.pack(side=tk.RIGHT, padx=5)

        self.save_report_button = ttk.Button(button_frame, text="Zapisz raport", command=self.generate_report, padding=(5, 1), state=tk.DISABLED)
        self.save_report_button.pack(side=tk.RIGHT, padx=5 )

        # === STATUS ===
        self.status_label = ttk.Label(button_frame, foreground='white', font=('Segoe UI', 10), background='navy', text="Gotowy")
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.progress = ttk.Progressbar(button_frame, orient="horizontal", length=150, mode="determinate")
        self.progress.pack(side=tk.LEFT, padx=10)
        
        # Nazwa modelu (jako label) po prawej stronie
        model_name = os.path.basename(self.model_path) if self.model_path else "USGEquina-Pregna_v1_0.keras"
        self.model_label = ttk.Label(button_frame, foreground='white', font=('Segoe UI', 10), background='navy', text=f"Model: {model_name}")
        self.model_label.pack(side=tk.LEFT, padx=10)
        
        self.model_label = ttk.Label(button_frame, foreground='white', font=('Segoe UI', 10), background='navy', text="ALK.BIZNES.AI.GR12.G2, 2025")
        self.model_label.pack(side=tk.RIGHT, padx=10)
        
        

        # === INICJALIZACJA STANU ===
        self.current_image_path = None
        self.analysis_result = None



    """
    def setup_gui(self):
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X)

        self.load_image_button = ttk.Button(top_frame, text="Wczytaj obraz", command=self.load_image)
        self.load_image_button.pack(side=tk.LEFT, padx=5)

        self.analyze_button = ttk.Button(top_frame, text="Analizuj", command=self.analyze_current_image, state=tk.DISABLED)
        self.analyze_button.pack(side=tk.LEFT, padx=5)

        self.save_report_button = ttk.Button(top_frame, text="Zapisz raport", command=self.generate_report, state=tk.DISABLED)
        self.save_report_button.pack(side=tk.LEFT, padx=5)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        ttk.Label(left_panel, text="Imię klaczy:").pack()
        self.klacz_name = tk.StringVar()
        ttk.Entry(left_panel, textvariable=self.klacz_name).pack()

        ttk.Label(left_panel, text="Wiek klaczy:").pack()
        self.klacz_age = tk.StringVar()
        ttk.Entry(left_panel, textvariable=self.klacz_age).pack()

        ttk.Label(left_panel, text="Szacowany dzień:").pack()
        self.estimated_day = tk.StringVar()
        ttk.Entry(left_panel, textvariable=self.estimated_day).pack()

        self.status_label = ttk.Label(main_frame, text="Gotowy")
        self.status_label.pack(fill=tk.X)

        self.progress = ttk.Progressbar(main_frame, orient="horizontal", length=200, mode="determinate")
        self.progress.pack(fill=tk.X)

        image_frame = ttk.Frame(content_frame)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.image_label = ttk.Label(image_frame, text="Obraz", background="gray")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(right_frame, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        self.current_image_path = None
        self.analysis_result = None
    """
    
    def load_models(self):
        def thread_load():
            from prediction import load_pregnancy_model, load_day_estimation_model
            self.pregnancy_model = load_pregnancy_model(self.model_path, self.log_file)
            self.day_model, self.day_mapping = load_day_estimation_model(log_file=self.log_file)
        threading.Thread(target=thread_load, daemon=True).start()
       
        
 

    def plot_features(self, features_dict):
        """Rysuje wykres słupkowy cech obrazu w ramce GUI"""
        # Usuń stare wykresy z ramki
        for widget in self.features_frame.winfo_children():
            widget.destroy()

        # Przygotowanie danych
        labels = [
            "Średnia intensywność", "Kontrast", "Entropia", 
            "Krawędzie", "Płyn (ratio)", "Tkanka (ratio)"
        ]
        values = [
            features_dict.get('mean_intensity', 0),
            features_dict.get('contrast', 0),
            features_dict.get('entropy', 0),
            features_dict.get('edge_magnitude', 0),
            features_dict.get('potential_fluid_ratio', 0),
            features_dict.get('potential_tissue_ratio', 0)
        ]

        # Tworzenie wykresu
        fig = plt.Figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        bars = ax.barh(labels, values, color='mediumseagreen', edgecolor='lightgreen')

        # Styl
        ax.set_xlim(0, max(values) * 1.2 if values else 1)
        ax.set_title("Cechy obrazu USG", fontsize=11, weight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.5)

        # Etykiety wartości
        for bar in bars:
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.2f}", va='center', fontsize=9)

        fig.tight_layout()

        # Umieszczenie w GUI
        canvas = FigureCanvasTkAgg(fig, master=self.features_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    """
    def plot_features(self, feature_dict):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        for widget in self.features_frame.winfo_children():
            widget.destroy()

        labels = ["Intens.", "Kontrast", "Entropia", "Płyn", "Tkanka"]
        values = [
            feature_dict.get("mean_intensity", 0),
            feature_dict.get("contrast", 0),
            feature_dict.get("entropy", 0),
            feature_dict.get("potential_fluid_ratio", 0),
            feature_dict.get("potential_tissue_ratio", 0),
        ]

        fig = plt.Figure(figsize=(4.5, 2.5), dpi=100)
        ax = fig.add_subplot(111)
        ax.bar(labels, values, color='skyblue')
        ax.set_ylim(0, 1)
        ax.set_title("Cechy obrazu")

        canvas = FigureCanvasTkAgg(fig, master=self.features_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    """

    def load_image(self):
        """Wczytuje obraz USG i przygotowuje do analizy."""
        file_path = filedialog.askopenfilename(
            title="Wybierz obraz USG",
            filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("Wszystkie pliki", "*.*")]
        )
        if not file_path:
            return

        try:
            self.current_image_path = file_path
            img = Image.open(file_path)
            img.thumbnail((600, 400), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)

            self.image_label.configure(image=tk_img, text="")
            self.image_label.image = tk_img

            self.analyze_button.config(state=tk.NORMAL)
            self.result_text.configure(state='normal')
            self.result_text.delete(1.0, tk.END)
            self.result_text.configure(state='disabled')

            self.update_status(f"Wczytano obraz: {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("Błąd wczytywania", f"Nie udało się wczytać obrazu:\n{e}")
            self.update_status("Błąd podczas wczytywania obrazu", error=True)



    def analyze_current_image(self):
        if not self.current_image_path:
            messagebox.showwarning("Brak obrazu", "Wczytaj obraz USG przed analizą.")
            return
        
        # Sprawdzenie danych klaczy
        name = self.klacz_name.get().strip()
        age = self.klacz_age.get().strip()
        day = self.estimated_day.get().strip()

        if not name or not age or not day:
            messagebox.showwarning("Brak danych", "Uzupełnij dane klaczy: imię, wiek oraz szacowany dzień cyklu.")
            return
        # Przypisanie danych
        additional_info = {
            'klacz_name': self.klacz_name.get(),
            'klacz_age': self.klacz_age.get(),
            'estimated_day': self.estimated_day.get()
        }

        try:
            # Wczytanie i przygotowanie obrazu
            img = tf.keras.preprocessing.image.load_img(self.current_image_path, target_size=IMAGE_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            # Predykcja
            prediction = self.pregnancy_model.predict(img_array)[0]
            confidence = float(prediction[0]) if isinstance(prediction, (list, np.ndarray)) else float(prediction)
            predicted_class = "pregnant" if confidence > 0.5 else "not_pregnant"

            # Analiza cech obrazu
            image_features = analyze_image_features(self.current_image_path, IMAGE_SIZE)

            # Generowanie opisu
            description = generate_description(
                predicted_class,
                confidence,
                image_path=self.current_image_path,
                additional_info=additional_info
            )

            # Zapisz wynik do stanu aplikacji
            self.analysis_result = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "description": description,
                "image_path": self.current_image_path,
                "additional_info": additional_info,
                "image_features": image_features
            }

            # Wyświetl opis w GUI
            self.result_text.configure(state='normal')
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"WYNIK PREDYKCJI: {predicted_class.upper()}\n")
            self.result_text.insert(tk.END, f"Pewność predykcji: {confidence*100:.2f}%\n\n")
            self.result_text.insert(tk.END, description)
            self.result_text.configure(state='disabled')

            # Wizualizacja cech
            if image_features:
                self.plot_features(image_features)

            # Aktywuj przycisk zapisu
            self.save_report_button.config(state=tk.NORMAL)
            self.update_status("Analiza zakończona pomyślnie")
            #Aktywuj przycisk legendy cech
            self.legend_button.config(state=tk.NORMAL)

        except Exception as e:
            log_error(f"Błąd predykcji: {e}", self.log_file)
            messagebox.showerror("Błąd analizy", f"Wystąpił błąd podczas analizy obrazu:\n{e}")
            self.update_status("Błąd podczas analizy", error=True)



    def generate_report(self):
        if not self.analysis_result:
            messagebox.showwarning("Brak danych", "Najpierw przeprowadź analizę obrazu.")
            return

        try:
            image_features = self.analysis_result.get("image_features")

            report_path = generate_pdf_report(
                image_path=self.analysis_result["image_path"],
                predicted_class=self.analysis_result["predicted_class"],
                confidence=self.analysis_result["confidence"],
                description=self.analysis_result["description"],
                additional_info=self.analysis_result.get("additional_info"),
                image_features=self.analysis_result.get("image_features")
            )

            if report_path and os.path.exists(report_path):
                messagebox.showinfo("Sukces", f"Raport zapisano: {report_path}")
                self.update_status("Raport wygenerowany pomyślnie")
            else:
                raise Exception("Raport nie został zapisany.")

        except Exception as e:
            log_error(f"Błąd generowania raportu: {e}", self.log_file)
            messagebox.showerror("Błąd", f"Wystąpił błąd podczas generowania raportu:\n{e}")
            self.update_status("Błąd podczas generowania raportu", error=True)
    
    def update_status(self, message, loading=False, error=False):
        """Aktualizuje pasek statusu"""
        self.status_label.config(text=message, foreground="red" if error else "white")
    
        if loading:
            self.progress.start()
        else:
            self.progress.stop()
            self.progress["value"] = 0 if error else 100
