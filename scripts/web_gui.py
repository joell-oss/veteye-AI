# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import os
import datetime
import numpy as np
import tensorflow as tf
import asyncio
import traceback  # Dodanie modułu do śledzenia błędów
from PIL import Image
from nicegui import ui, app
from image_analysis import analyze_image_features
from diagnostic_utils import generate_description, generate_pdf_report, save_features_plot_to_file
from fastapi.staticfiles import StaticFiles

'''
Sekcja inicjalizująca stałe, wczytująca wytrenowany model oraz konfigurująca katalogi wyjściowe.
Zmienna IMAGE_SIZE określa docelowy rozmiar obrazów wejściowych do modelu.
MODEL_PATH wskazuje na ścieżkę do zapisanego modelu sieci neuronowej.
Katalogi na raporty i wyniki tworzone są automatycznie, jeśli nie istnieją.
Zmienne globalne przechowują m.in. ścieżkę do wczytanego obrazu oraz edytowany opis diagnostyczny.
Lista class_names zawiera etykiety klas rozpoznawanych przez model.
Dodane style CSS zwiększają wysokość pola diagnostycznego oraz poprawiają widoczność i ergonomię przycisków pływających i stopki w interfejsie użytkownika.
'''

# Stałe
IMAGE_SIZE = (380, 380)
MODEL_PATH = '../checkpoints/USGEquina-Pregna_v1_0.keras'
REPORTS_DIR = 'raporty'
MANUAL_DIR = 'wyniki'
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MANUAL_DIR, exist_ok=True)


# Zmienne globalne
uploaded_image_path = None
edited_description = ''
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['not_pregnant', 'pregnant']

# Style CSS dla większej wysokości pola diagnostycznego i przycisków pływających
ui.add_head_html("""
<style>
.diagnostic-textarea {
    min-height: 3000px !important;
    height: 70vh !important; /* 70% wysokości widocznego obszaru */
}

.floating-buttons {
    position: fixed;
    bottom: 80px; /* Zwiększamy wartość, aby przyciski nie nachodziły na stopkę */
    right: 20px;
    display: flex;
    gap: 10px;
    z-index: 1000;
}

/* Style dla przycisków pływających - nie okrągłe, ale standardowe */
.floating-buttons .q-btn {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
}

.floating-buttons .q-btn:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
}

/* Styl dla stopki */
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    z-index: 900;
}
</style>
""")

# --- Analiza obrazu ---
async def analyze_image():
    '''
    Funkcja analyze_image realizuje kompletny proces analizy wczytanego obrazu USG.
    Sprawdza poprawność wprowadzonych danych, blokuje interfejs na czas analizy i wyświetla odpowiednie powiadomienia dla użytkownika.
    Obraz jest przetwarzany przez wytrenowany model, na podstawie predykcji generowany jest opis i wykres cech diagnostycznych.
    Wyniki analizy są prezentowane w interfejsie oraz automatycznie zapisywane do pliku PDF.
    Po zakończeniu analizy interfejs jest odblokowywany, a użytkownik otrzymuje informację o zakończeniu operacji.
    '''

    global edited_description
    analysis_notification = None
    
    if not uploaded_image_path:
        ui.notify('Najpierw załaduj obraz.', color='negative')
        return
    if not mare_name.value or not mare_age.value or not estimated_day.value:
        ui.notify('Uzupełnij dane klaczy przed analizą.', color='warning')
        return
    
    try:
        # Blokada przycisków podczas analizy
        analyze_float_btn.disable()
        save_float_btn.disable()
        about_float_btn.disable()
        legend_float_btn.disable()
        visit_float_btn.disable()
        
        # Komunikat dla użytkownika
        try:
            analysis_notification = ui.notify("Generowanie opisu obrazu ultrasonograficznego przez AI...", 
                    spinner=True,
                    closable=False,  # Nie pozwalamy zamknąć powiadomienia
                    timeout=10)  # Bez automatycznego zamknięcia
        except Exception as notify_error:
            print(f"Błąd powiadomienia: {notify_error}")
            analysis_notification = None
        
        # Informacja o trwającej analizie w polu wyników
        result_area.value = "Trwa analiza obrazu... Proszę czekać."
        await asyncio.sleep(0.3)  
        
        img = tf.keras.preprocessing.image.load_img(uploaded_image_path, target_size=IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        confidence = float(prediction[0]) if isinstance(prediction, (list, np.ndarray)) else float(prediction)
        predicted_class = class_names[1] if confidence > 0.5 else class_names[0]

        info = {
            'klacz_name': mare_name.value,
            'klacz_age': mare_age.value,
            'estimated_day': estimated_day.value
        }

        image_features = analyze_image_features(uploaded_image_path, IMAGE_SIZE)
        description = generate_description(predicted_class, confidence, image_path=uploaded_image_path, additional_info=info)
        edited_description = f"WYNIK PREDYKCJI: {predicted_class.upper()}\nPewność: {confidence*100:.2f}%\n\n{description}"
        
        # Aktualizacja pola wyników
        result_area.value = edited_description
        await asyncio.sleep(0.5)  

        # Wykres cech
        try:
            # Własna wersja funkcji do tworzenia wykresu z mniejszą czcionką
            def create_features_chart(image_features):
                import matplotlib.pyplot as plt
                import tempfile
                
                # Standardowy rozmiar figury
                fig, ax = plt.subplots(figsize=(6, 4))
                
                # Konfiguracja czcionki matplotlib - zmniejszone rozmiary
                plt.rcParams.update({
                    'font.size': 8,        # Podstawowy rozmiar czcionki
                    'axes.titlesize': 10,  # Rozmiar tytułów osi
                    'axes.labelsize': 9,   # Rozmiar etykiet osi
                    'xtick.labelsize': 8,  # Rozmiar etykiet na osi X
                    'ytick.labelsize': 8,  # Rozmiar etykiet na osi Y
                })
                
                labels = list(image_features.keys())
                values = list(image_features.values())
                
                # kolory wykresu
                bar_color = '#90ee90'  # Jasnozielony
                edge_color = '#66bb66'  # Ciemniejszy zielony dla krawędzi
                
                # wykres słupkowy
                bars = ax.barh(labels, values, color=bar_color, edgecolor=edge_color, height=0.5)
                
                # etykiety wartości po prawej stronie słupków - mniejsza czcionka
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                            va='center', fontsize=7)
                
                # Tytuł wykresu
                ax.set_title("Cechy obrazu USG", fontsize=10, fontweight='bold')
                
                # Zakresy osi
                ax.set_xlim([0, max(values) * 1.10])  # 10% dodatkowej przestrzeni dla etykiet
                
                # Ustawienia siatki
                ax.grid(axis='x', linestyle='--', alpha=0.5)
                
                # Układ wykresu - kompaktowy
                plt.tight_layout(pad=0.5)
                
                # Zapisujemy wykres do pliku tymczasowego
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                plt.savefig(tmp_file.name, dpi=100)
                plt.close(fig)
                
                return tmp_file.name
                
            # Używamy własnej funkcji z mniejszą czcionką
            chart_path = create_features_chart(image_features)
            chart_display.set_source(chart_path)
            
            # odświeżenie komponentów UI poprzez dłuższy sleep
            await asyncio.sleep(0.5)  
        except Exception as chart_err:
            ui.notify(f'Błąd wygenerowania wykresu: {chart_err}', color='warning')
            print(f"Błąd wykresu: {chart_err}")

        # Auto zapis - używamy poprawnej sygnatury funkcji z diagnostic_utils.py
        report_path = generate_pdf_report(uploaded_image_path, predicted_class, confidence, description, 
                                        additional_info=info, image_features=image_features)

        # Zamykamy powiadomienie o trwającej analizie
        try:
            if analysis_notification is not None:
                analysis_notification.close()
                analysis_notification = None
        except Exception as close_error:
            print(f"Błąd zamykania powiadomienia: {close_error}")
        
        # Dajemy czas na zakończenie operacji zapisu
        await asyncio.sleep(0.5)

        # Powiadomienie o zakończeniu
        ui.notify(f'Analiza zakończona. Raport PDF zapisany automatycznie.', 
                 color='positive',
                 timeout=5)  # Automatycznie zniknie po 5 sekundach
        
        # Dialog informujący o zakończeniu analizy
        with ui.dialog() as complete_dialog:
            with ui.card():
                ui.label('Analiza zakończona').classes('text-xl font-bold')
                ui.label('Raport PDF został zapisany automatycznie.')
                ui.button('Zamknij', on_click=complete_dialog.close).props('color=primary')
        complete_dialog.open()

    except Exception as e:
        print(f"Błąd analizy: {e}")
        traceback.print_exc()  # Drukuje pełny stack trace błędu
        ui.notify(f'Błąd: {e}', color='negative')
    finally:
        # Odblokowujemy przyciski niezależnie od wyniku
        analyze_float_btn.enable()
        save_float_btn.enable()
        about_float_btn.enable()
        legend_float_btn.enable()
        visit_float_btn.enable()
        
        # Dajemy czas na zaktualizowanie interfejsu
        await asyncio.sleep(0.5)

# --- Zapis manualny PDF ---
async def manual_save_to_file():
    '''
    Funkcja manual_save_to_file umożliwia ręczne zapisanie raportu diagnostycznego w formacie PDF.
    Blokuje interfejs użytkownika na czas operacji zapisu oraz wyświetla stosowne powiadomienia o postępie i zakończeniu procesu.
    Uwzględnia aktualną treść opisu, w tym wszystkie zmiany wprowadzone przez użytkownika.
    Po zakończeniu zapisu odblokowuje przyciski i informuje użytkownika o sukcesie operacji lub błędach.
    '''
    notification = None
    global edited_description
    if not uploaded_image_path:
        ui.notify('Brak obrazu do zapisania.', color='warning')
        return
    
    try:
        # Blokujemy przyciski podczas zapisu
        analyze_float_btn.disable()
        save_float_btn.disable()
        about_float_btn.disable()
        legend_float_btn.disable()
        visit_float_btn.disable()
        
        # Powiadomienie o zapisie z timeoutem
        try:
            save_notification = ui.notify("Zapisywanie raportu do PDF...", 
                    spinner=True,
                    closable=False,
                    timeout=5)  # Bez automatycznego zamknięcia
        except Exception as notify_error:
            print(f"Błąd powiadomienia: {notify_error}")
            save_notification = None
        
        # Krótka pauza, aby UI mogło się zaktualizować
        await asyncio.sleep(0.3)
        
        # Pobieramy aktualną treść z pola tekstowego - uwzględnia zmiany wprowadzone przez użytkownika
        current_text = result_area.value
        
        info = {
            'klacz_name': mare_name.value,
            'klacz_age': mare_age.value,
            'estimated_day': estimated_day.value
        }
        image_features = analyze_image_features(uploaded_image_path, IMAGE_SIZE)
        predicted_class = 'pregnant' if 'PREGNANT' in current_text.upper() else 'not_pregnant'
        confidence_line = [line for line in current_text.split('\n') if 'Pewność:' in line]
        confidence = float(confidence_line[0].split(':')[1].strip('% \n')) / 100 if confidence_line else 0.85

        # Używamy poprawnej sygnatury funkcji z diagnostic_utils.py z aktualną treścią
        report_path = generate_pdf_report(uploaded_image_path, predicted_class, confidence, current_text, 
                                         additional_info=info, image_features=image_features)

        # Zamykamy powiadomienie o zapisie
        try:
            if save_notification is not None:
                save_notification.close()
        except Exception as close_error:
            print(f"Błąd zamykania powiadomienia: {close_error}")

        # Powiadomienie o zakończeniu z krótkim czasem trwania
        ui.notify(f'Raport PDF zapisany ręcznie.', 
                 color='positive',
                 timeout=5)  # Automatycznie zniknie po 5 sekundach
                 
        # Dialog informujący o zakończeniu zapisu
        with ui.dialog() as complete_dialog:
            with ui.card():
                ui.label('Zapis ukończony').classes('text-xl font-bold')
                ui.label('Raport PDF został zapisany ręcznie.')
                ui.button('Zamknij', on_click=complete_dialog.close).props('color=primary')
        complete_dialog.open()

    except Exception as e:
        ui.notify(f'Błąd zapisu: {e}', color='negative')
    finally:
        # Odblokowujemy przyciski niezależnie od wyniku
        analyze_float_btn.enable()
        save_float_btn.enable()
        about_float_btn.enable()
        legend_float_btn.enable()
        visit_float_btn.enable()
        
        # Zamykamy powiadomienie jeśli jeszcze istnieje
        if notification is not None:
            try:
                notification.close()
            except:
                pass

# --- Legenda cech ---
def feature_legend():
    '''
    Funkcja feature_legend zwraca opis wybranych cech obrazu USG w formacie HTML.
    Wyjaśnia znaczenie każdej cechy oraz podaje jej typowy zakres wartości dla dobrej jakości diagnostycznej.
    Pomaga użytkownikowi w interpretacji wykresów i wyników analizy obrazów ultrasonograficznych.
    '''

    return """
        <div style="font-size: 14px; line-height: 1.5;">
        • <b>mean_intensity</b> – średnia jasność pikseli obrazu.<br/>
        Typowy zakres dobrej jakości: 0.2 – 0.5.<br/>
        Znaczenie diagnostyczne: Zbyt niska = niedoświetlony obraz; zbyt wysoka = prześwietlony.<br/><br/>
        
        • <b>std_intensity</b> – odchylenie standardowe jasności (różnorodność jasności).<br/>
        Typowy zakres dobrej jakości: 0.1 – 0.3.<br/>
        Znaczenie diagnostyczne: Za niska = obraz jednolity (mało informacji); za wysoka = szum.<br/><br/>
        
        • <b>contrast</b> – stosunek kontrastu do jasności (czy struktury są wyraźne).<br/>
        Typowy zakres dobrej jakości: > 0.4.<br/>
        Znaczenie diagnostyczne: Wyraźne różnice między strukturami (czytelność).<br/><br/>
        
        • <b>entropy</b> – miara złożoności tekstury (szum lub szczegóły).<br/>
        Typowy zakres dobrej jakości: 2.5 – 4.5.<br/>
        Znaczenie diagnostyczne: Za niska = mało szczegółów; za wysoka > 5.0 = zbyt duży szum.<br/><br/>
        
        • <b>edge_magnitude</b> – siła krawędzi (liczba i intensywność konturów).<br/>
        Typowy zakres dobrej jakości: 0.2 – 0.5.<br/>
        Znaczenie diagnostyczne: Zbyt niska = mało struktur; za wysoka = szum lub artefakty.<br/><br/>
        
        • <b>potential_fluid_ratio</b> – procent jasnych obszarów sugerujących płyn.<br/>
        Typowy zakres dobrej jakości: 0.05 – 0.3.<br/>
        Znaczenie diagnostyczne: Więcej = prawdopodobieństwo obecności pęcherzyka.<br/><br/>
        
        • <b>potential_tissue_ratio</b> – procent ciemnych obszarów sugerujących tkanki.<br/>
        Typowy zakres dobrej jakości: 0.4 – 0.8.<br/>
        Znaczenie diagnostyczne: Zbyt mało = słaba widoczność struktur anatomicznych.
        </div>
    """

'''
Kod odpowiada za budowę głównego interfejsu użytkownika systemu diagnostyki USG klaczy.
Zawiera pasek nagłówka, formularz do wprowadzania danych badania oraz moduł do wczytywania obrazów USG.
W centralnej części aplikacji wyświetlane są jednocześnie: obraz ultrasonograficzny oraz wykres cech diagnostycznych.
Interfejs został zaprojektowany z myślą o przejrzystości i wygodzie użytkownika (UX/UI).
'''

# Pasek nagłówka
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
static_path = os.path.join(project_root, 'static')
app.mount('/static', StaticFiles(directory=static_path), name='static')

with ui.header().classes('bg-blue-900 px-4 py-2'):
    with ui.row().classes('w-full items-center justify-between'):

        # Lewa strona
        ui.label("veteye.AI - System diagnostyki USG klaczy") \
            .classes('text-white text-lg font-bold')

        # Prawa strona
        with ui.row().classes('items-center gap-3'):
            # Login tekst
            ui.label('LoginID: 0658793211 (Państwowa Stadnina Koni Hodowlanych - Niejanów Niepodlaski)') \
                .classes('text-white text-xs')

            ui.icon('account_circle') \
                .classes('text-white cursor-pointer') \
                .style('font-size: 40px; height: 40px; width: 40px;') \
                .tooltip('Konto i ustawienia')

            with ui.link(target='https://tu2.app/', new_tab=True):
                ui.image('/static/images/logo.png') \
                    .classes('transition-all duration-300 hover:scale-105') \
                    .style('height: 40px; max-width: 180px; width: 160px;')


# --- GÓRNY UKŁAD: dane klaczy + wczytaj obraz ---
with ui.row().classes('w-full items-end gap-4 mt-4'):
    mare_name = ui.input(label='Imię klaczy').props('outlined dense').classes('w-1/4')
    mare_age = ui.input(label='Wiek klaczy').props('outlined dense').classes('w-1/4')
    estimated_day = ui.input(label='Szacowany dzień').props('outlined dense').classes('w-1/4')
    with ui.column().classes('w-1/4'):
        upload = ui.upload(label='Wczytaj obraz', auto_upload=True, multiple=False).classes('w-full')
        image_filename = ui.label("Nie wczytano pliku").style('font-size: 12px; color: gray')

# --- ŚRODKOWY UKŁAD: obraz i wykres obok siebie ---
with ui.grid(columns=2).classes('w-full gap-4 mt-2'):
    with ui.card().classes('col-span-1'):
        ui.label("Obraz USG").classes('text-lg font-bold')
        image_display = ui.image().style('width: 100%; height: auto; border: 1px solid #ccc')
    
    with ui.card().classes('col-span-1'):
        ui.label("Wykres cech obrazu").classes('text-lg font-bold')
        chart_display = ui.image().style('width: 100%; height: auto; border: 1px solid #ccc')

# --- Przeniesienie definicji funkcji show_legend i show_about, które zostały usunięte wraz z przyciskami ---
def show_legend():
    '''
    Funkcja show_legend wyświetla okno dialogowe z legendą opisującą cechy obrazu USG.
    Pozwala użytkownikowi w łatwy sposób zapoznać się ze znaczeniem poszczególnych parametrów diagnostycznych.
    Okno można zamknąć przyciskiem.
    '''

    with ui.dialog().classes('w-1/2') as dialog:
        with ui.card():
            ui.label('LEGENDA CECH OBRAZU').classes('text-xl font-bold')
            ui.html(feature_legend())
            ui.button('Zamknij', on_click=dialog.close).props('color=primary')
    dialog.open()

def show_consent():
    '''
    Funkcja `show_consent()` tworzy i wyświetla okno dialogowe z treścią zgody na przetwarzanie danych osobowych 
    oraz medycznych w kontekście wykorzystania systemu diagnostycznego opartego na sztucznej inteligencji (AI) 
    do oceny stanu zdrowia klaczy.
    Zastosowanie:
    Funkcja ta może być użyta w aplikacji webowej do przedstawienia użytkownikowi obowiązkowej zgody, 
    np. przed uruchomieniem procedury diagnostycznej z wykorzystaniem AI.
    '''
    
    dialog = ui.dialog()
    with dialog:
        with ui.card().classes('p-4'):
            ui.label('ZGODA NA PRZETWARZANIE DANYCH').classes('text-xl font-bold mb-2')
            ui.html("""<div style="font-size: 12px; line-height: 1.4;">
                Wyrażam zgodę na przetwarzanie danych osobowych, w tym danych identyfikujących zwierzę oraz danych medycznych (obrazów ultrasonograficznych), przez operatora systemu diagnostycznego wykorzystującego sztuczną inteligencję (AI), w celu przeprowadzenia wspomaganej komputerowo oceny stanu zdrowia klaczy.<br><br>

                Oświadczam, że zostałem/-am poinformowany/-a, iż:<br><br>

                • Dane będą przetwarzane zgodnie z Rozporządzeniem Parlamentu Europejskiego i Rady (UE) 2016/679 (RODO) i aktami towarzyszącymi,<br>
                • a także z ustawą z dnia 21 grudnia 1990 r. o zawodzie lekarza weterynarii i izbach lekarsko-weterynaryjnych.<br>
                • Przetwarzanie danych odbywa się w celach diagnostycznych oraz doskonalenia jakości usług świadczonych przez lekarzy weterynarii.<br>
                • Dane nie będą udostępniane osobom trzecim bez wyraźnej podstawy prawnej lub odrębnej zgody.<br><br>

                Lekarz weterynarii nadzorujący proces diagnostyczny ponosi pełną odpowiedzialność za interpretację wyników oraz ma obowiązek kierować się zasadami Kodeksu Etyki Lekarza Weterynarii, w tym zachowaniem tajemnicy zawodowej.<br><br>

                <b>Mam prawo do:</b><br>
                – dostępu do swoich danych,<br>
                – ich sprostowania, usunięcia lub ograniczenia przetwarzania,<br>
                – wniesienia sprzeciwu,<br>
                – cofnięcia zgody w dowolnym momencie bez wpływu na legalność przetwarzania dokonanego przed jej cofnięciem.<br><br>

                ✅ Zgadzam się na przetwarzanie danych zgodnie z powyższą klauzulą.<br><br>

                <i>Oświadczenie o zgodzie na wykorzystanie systemu sztucznej inteligencji (AI) zostało złożone przez Państwową Stadninę Koni Hodowlanych - Niejanów Niepodlaski, zwaną dalej Subskrybentem Systemu, w dniu 15 czerwca 2025 r.</i>
                </div>
            """)
            ui.button('Zamknij', on_click=dialog.close).props('color=primary')
    dialog.open()

def show_about():
    '''
    Kod definiuje elementy interfejsu odpowiedzialne za prezentację informacji o systemie, 
    obsługę opisu diagnostycznego oraz główne przyciski funkcyjne. 
    Funkcja show_about wyświetla szczegółowe informacje o projekcie, autorach i opiekunach pracy. 
    Pole tekstowe result_area umożliwia wygodne przeglądanie i edycję opisu diagnostycznego. 
    Na dole ekranu znajdują się pływające przyciski umożliwiające szybki dostęp do najważniejszych funkcji aplikacji.
    '''
    with ui.dialog().classes('w-2/3') as dialog:
        with ui.card():
            ui.label('O SYSTEMIE').classes('text-xl font-bold')
            ui.html("""
                <div style="font-size: 16px; text-align: center; padding: 20px;">
                    <p><b>KOZMINSKI EXECUTIVE BUSINESS SCHOOL CENTRUM DORADZTWA I KSZTAŁCENIA MENEDŻERÓW</b></p>
                    <p style="margin-top: 15px;">Studia podyplomowe: Biznes.ai: zarządzanie projektami sztucznej inteligencji, edycja 12</p>
                    <p style="margin-top: 15px;">Wdrożenie platformy wsparcia weterynaryjnej diagnostyki USG z wykorzystaniem AI</p>
                    <p style="margin-left: 15px;">Autorzy :</p>
                    <p style="margin-right: 15px;">Jan Polański - nr albumu 67192-CKP</p>
                    <p style="margin-right: 15px;">Paweł Rusek - nr albumu 67193-CKP</p>
                    <p style="margin-right: 15px;">Józef Sroka - nr albumu 67195-CKP</p>
                    <p style="margin-right: 15px;">Krzysztof Trawiński - nr albumu 67201-CKP</p>
                    <p style="margin-right: 15px;">Iwona Grub-Malinowska - nr albumu 67409-CKP</p>
                    <p style="margin-right: 15px;">Adam Lasko - nr albumu 67182-CKP</p>
                    <p style="margin-left: 15px;">Praca pisana pod kierunkiem:</p>
                    <p style="margin-right: 15px;">Mec. Romana Biedy</p>
                    <p style="margin-right: 15px;">Tomasza Klekowskiego</p>
                    <p style="margin-right: 15px;">Andrzeja Jankowskiego</p>
                    <p style="margin-top: 15px;">System diagnostyki USG klaczy</p>
                    <p style="margin-top: 15px;">wersja 1.01.001</p>
                    <p style="margin-top: 15px;">&copy; czerwiec 2025</p>
                </div>
            """)
            ui.button('Zamknij', on_click=dialog.close).props('color=primary')
    dialog.open()

# --- OPIS DIAGNOSTYCZNY --- z powiększoną wysokością i responsywnością
result_area = ui.textarea(label='Opis diagnostyczny').props('outlined').classes('w-full mt-4 diagnostic-textarea').props('rows=40')

# --- PRZYCISKI PŁYWAJĄCE NA DOLE EKRANU ---
with ui.element('div').classes('floating-buttons'):
    about_float_btn = ui.button('O SYSTEMIE', on_click=show_about).props('color=secondary icon=info')
    consent_float_btn = ui.button('ZGODY I OŚWIADCZENIA', on_click=show_consent).props('color=secondary icon=verified_user')
    visit_float_btn = ui.button('UMÓW WIZYTĘ', on_click=lambda: ui.notify("Funkcja rejestracji wizyty będzie dostępna w pełnej wersji.")).props('color=primary icon=calendar_month')
    save_float_btn = ui.button('ZAPISZ DO PLIKU', on_click=manual_save_to_file).props('color=primary icon=save')
    legend_float_btn = ui.button('LEGENDA CECH', on_click=show_legend).props('color=secondary icon=help_outline')
    analyze_float_btn = ui.button('ANALIZUJ OBRAZ', on_click=analyze_image).props('color=primary icon=search')

# --- Obsługa wczytywania obrazu ---
'''
Kod odpowiada za obsługę procesu wczytywania obrazu do aplikacji oraz prezentację pliku w interfejsie.
Po załadowaniu plik jest zapisywany na serwerze i automatycznie wyświetlany użytkownikowi.
Dodatkowo sekcja definiuje stopkę z informacją o modelu i wersji systemu.
Ostatnia linia uruchamia aplikację w trybie demonstracyjnym z odpowiednim tytułem.
'''

@upload.on_upload
def handle_upload(e):
    global uploaded_image_path
    if not e.content:
        ui.notify('Brak zawartości pliku')
        return
    file_name = e.name
    uploaded_file = e.content
    uploaded_image_path = os.path.join('static/uploaded_images', file_name)
    os.makedirs(os.path.dirname(uploaded_image_path), exist_ok=True)
    with open(uploaded_image_path, 'wb') as f:
        f.write(uploaded_file.read())
    image_display.set_source(uploaded_image_path)
    image_filename.text = f"Plik: {file_name}"
    ui.notify('Załadowano obraz')

# Stopka
with ui.footer().classes('bg-blue-800 text-white p-3'):
    with ui.row().classes('w-full justify-between items-center'):
        ui.label(f"Model: {os.path.basename(MODEL_PATH)}").style('font-size: 14px;')
        ui.label(f"© 2025 veteye.AI | wersja 1.01.001").style('font-size: 14px;')

# --- Uruchom GUI ---
ui.run(title="veteye.AI - Web GUI - demonstrator")
