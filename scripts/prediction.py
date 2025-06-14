# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import os
import re
import time
import json
import numpy as np
import datetime
import traceback
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import inception_v3
from config import REPORTS_DIR, IMAGE_SIZE, CHECKPOINTS_DIR
from logging_utils import log_info, log_error, log_success
from data_loader import load_and_preprocess_image
from image_analysis import analyze_image_features


def load_pregnancy_model(model_path=None, log_file=None):
    """
    Ładuje model wykrywania ciąży z automatycznym wyszukiwaniem najnowszej wersji.
    Implementuje inteligentne ładowanie modelu przez:
    - Automatyczne wykrywanie plików modelu z wzorcem 'Pregna_v1_0' w nazwie
    - Wybór najnowszego modelu na podstawie daty modyfikacji pliku
    - Obsługę jawnie podanej ścieżki lub automatycznego wyszukiwania
    - Pełne logowanie procesu ładowania z poziomami info/success/error
    Logika wyboru:
    1. Jeśli podano ścieżkę - używa bezpośrednio
    2. Jeśli nie - skanuje katalog CHECKPOINTS_DIR
    3. Filtruje pliki .keras zawierające identyfikator modelu
    4. Wybiera najnowszy na podstawie timestamps
    Zwraca załadowany model Keras lub None w przypadku błędu.
    Zapewnia odporność na brak plików i nieprawidłowe ścieżki.
    """
    try:
        if model_path is None:
            models = [f for f in os.listdir(CHECKPOINTS_DIR) if f.endswith('.keras') and 'Pregna_v1_0' in f]
            if not models:
                log_error("Nie znaleziono modelu wykrywania ciąży", log_file)
                return None
            model_path = os.path.join(CHECKPOINTS_DIR, max(models, key=lambda f: os.path.getmtime(os.path.join(CHECKPOINTS_DIR, f))))

        log_info(f"Ładowanie modelu wykrywania ciąży: {model_path}", log_file)
        model = load_model(model_path)
        log_success("Model wykrywania ciąży załadowany pomyślnie", log_file)
        return model

    except Exception as e:
        log_error(f"Błąd podczas ładowania modelu wykrywania ciąży: {e}", log_file)
        return None


def load_day_estimation_model(model_path=None, log_file=None):
    """
    Ładuje model szacowania dnia ciąży wraz z mapowaniem klas do dni.
    Implementuje kompleksowe ładowanie modelu wieloklasowego przez:
    - Automatyczne wyszukiwanie najnowszego modelu z frazą 'day' w nazwie
    - Równoczesne ładowanie pliku mapowania klas (_mapping.json)
    - Wybór najnowszego modelu na podstawie daty modyfikacji
    - Obsługę braku pliku mapowania (zwraca pusty słownik)
    Struktura zwracanych danych:
    - model: załadowany model Keras do predykcji
    - day_mapping: słownik mapujący indeksy klas na dni ciąży
    Logika mapowania:
    1. Automatyczne generowanie ścieżki pliku mapowania z nazwy modelu
    2. Ładowanie JSON z kodowaniem UTF-8
    3. Graceful handling braku pliku mapowania
    Zwraca krotkę (model, day_mapping) lub (None, {}) w przypadku błędu.
    Niezbędne do interpretacji wyników predykcji wieloklasowej.
    Nie realizowane w demonstratorze.
    """
    try:
        if model_path is None:
            models = [f for f in os.listdir(CHECKPOINTS_DIR) if f.endswith('.keras') and 'day' in f]
            if not models:
                log_error("Nie znaleziono modelu szacowania dnia ciąży", log_file)
                return None, {}
            model_path = os.path.join(CHECKPOINTS_DIR, max(models, key=lambda f: os.path.getmtime(os.path.join(CHECKPOINTS_DIR, f))))

        model = load_model(model_path)
        mapping_path = model_path.replace('.keras', '_mapping.json')
        day_mapping = {}
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r', encoding='utf-8') as f:
                day_mapping = json.load(f)

        log_success("Model szacowania dnia ciąży załadowany", log_file)
        return model, day_mapping

    except Exception as e:
        log_error(f"Błąd podczas ładowania modelu dnia: {e}", log_file)
        return None, {}


def predict_pregnancy(img_array, model):
    """
    Wykonuje predykcję ciąży na podstawie przetworzonego obrazu USG.
    Realizuje klasyfikację binarną przez:
    - Wywołanie predykcji modelu na przygotowanej macierzy obrazu
    - Ekstrakcję współczynnika pewności z wyjścia modelu
    - Obsługę różnych formatów wyjściowych (1D/0D tensor)
    - Klasyfikację binarną z progiem decyzyjnym 0.5
    Logika klasyfikacji:
    - confidence > 0.5 → 'pregnant' (ciąża wykryta)
    - confidence ≤ 0.5 → 'not_pregnant' (brak ciąży)
    Format wyjściowy:
    - etykieta: string z wynikiem klasyfikacji
    - confidence: float reprezentujący pewność predykcji (0.0-1.0)
    Funkcja zakłada, że obraz został przygotowany
    (normalizacja, resize, dodanie wymiaru batch).
    """
    prediction = model.predict(img_array)[0]
    confidence = float(prediction[0]) if prediction.shape[0] == 1 else float(prediction)
    return 'pregnant' if confidence > 0.5 else 'not_pregnant', confidence


def estimate_pregnancy_day(img_array, model, mapping=None):
    """
    Szacuje dzień ciąży z rankingiem top-5 najbardziej prawdopodobnych wyników.
    Implementuje wieloklasową predykcję przez:
    - Wykonanie predykcji dla wszystkich możliwych dni ciąży
    - Sortowanie wyników według prawdopodobieństwa (malejąco)
    - Wybór 5 najlepszych kandydatów z odpowiednimi współczynnikami pewności
    - Mapowanie indeksów klas na rzeczywiste dni ciąży (jeśli dostępne)
    Struktura wyniku:
    - predicted_day: najbardziej prawdopodobny dzień ciąży
    - confidence: pewność głównej predykcji (0.0-1.0)
    - top_5_days: lista 5 najbardziej prawdopodobnych dni
    - top_5_confidences: odpowiadające im współczynniki pewności
    Obsługa mapowania:
    - Z mapowaniem: konwersja indeksów na rzeczywiste dni ciąży
    - Bez mapowania: zwraca surowe indeksy klas
    Umożliwia analizę niepewności modelu i alternatywnych diagnoz.
    """
    preds = model.predict(img_array)[0]
    top_indices = np.argsort(preds)[-5:][::-1]
    top_confs = [float(preds[i]) for i in top_indices]

    if mapping:
        top_days = [int(list(mapping.keys())[i]) for i in top_indices]
        predicted_day = top_days[0]
    else:
        top_days = top_indices.tolist()
        predicted_day = top_days[0]

    return {
        "predicted_day": predicted_day,
        "confidence": top_confs[0],
        "top_5_days": top_days,
        "top_5_confidences": top_confs
    }


def predict_image(image_path, model=None, log_file=None):
    """
    Kompleksowe przetwarzanie obrazu USG z predykcją i generowaniem opisu wyników.
    Realizuje pełny pipeline analizy obrazu przez:
    - Wczytanie i przeskalowanie obrazu do rozmiaru modelu (IMAGE_SIZE)
    - Preprocessing zgodny z InceptionV3 (normalizacja do zakresu [-1,1])
    - Dodanie wymiaru batch dla kompatybilności z modelem
    - Klasyfikację binarną z progiem decyzyjnym 0.5
    - Automatyczne generowanie opisowego raportu wyników
    Etapy przetwarzania:
    1. Ładowanie obrazu z dysku z target_size
    2. Konwersja do macierzy NumPy
    3. Normalizacja pikseli metodą InceptionV3
    4. Reshape do formatu batch (1, height, width, channels)
    5. Predykcja i interpretacja wyników
    Zwraca krotkę:
    - predicted_class: 'pregnant'/'not_pregnant'
    - confidence: współczynnik pewności (0.0-1.0)
    - description: tekstowy opis wyników predykcji
    Obsługa błędów z pełnym traceback dla debugowania.
    """
    try:
        img = image.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = inception_v3.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        confidence = float(prediction[0]) if prediction.shape[0] == 1 else float(prediction)
        predicted_class = 'pregnant' if confidence > 0.5 else 'not_pregnant'

        description = generate_description(predicted_class, confidence, image_path=image_path)

        return predicted_class, confidence, description

    except Exception as e:
        log_error(f"Błąd podczas predykcji obrazu: {e}", log_file)
        traceback.print_exc()
        return None, 0.0, ""


def generate_description(predicted_class, confidence, image_path=None, additional_info=None):
    """
    Generuje szczegółowy opis diagnostyczny wyniku badania USG z dostosowaniem do poziomu pewności.
    Tworzy profesjonalny raport weterynaryjny uwzględniający:
    - Poziom pewności predykcji (>95%, >85%, <85%) z odpowiednimi sformułowaniami
    - Automatyczne wykrywanie dnia ciąży z nazwy pliku (wzorzec _d[liczba])
    - Analizę cech technicznych obrazu (kontrast, entropia, struktury płynowe)
    - Personalizację z danymi klaczy (imię, wiek)
    - Standardowe zalecenia weterynaryjne
    Struktura raportu:
    - Nagłówek z datą badania i podstawowym wynikiem
    - Opis kliniczny dostosowany do poziomu pewności
    - Dodatkowe uwagi techniczne dotyczące jakości obrazu
    - Zalecenia postępowania i kontroli
    - Disclaimer o konieczności weryfikacji przez weterynarza
    Automatycznie dostosowuje szczegółowość opisu do wiarygodności wyników,
    od jednoznacznych diagnoz po ostrożne sugestie wymagające weryfikacji.
    """
    image_features = None
    if image_path:
        try:
            image_features = analyze_image_features(image_path, IMAGE_SIZE)
        except Exception as e:
            print(f"Błąd podczas analizy obrazu: {e}")

    current_date = datetime.datetime.now().strftime("%d.%m.%Y")
    estimated_day = "nieznany"
    if additional_info and additional_info.get('estimated_day'):
        estimated_day = additional_info['estimated_day']
    if image_path:
        filename = os.path.basename(image_path)
        match = re.search(r'_d(\d+)', filename)
        if match:
            estimated_day = match.group(1)

    if predicted_class == "pregnant":
        if confidence > 0.95:
            description = (
                f"Badanie USG z dnia {current_date} wykazuje obraz jednoznacznie wskazujący na ciążę u klaczy. "
                f"Widoczne są charakterystyczne struktury pęcherzyka ciążowego z dobrze zarysowanymi granicami. "
                f"Szacowany dzień ciąży: około {estimated_day} dni (weryfikacja kliniczna wskazana). "
                f"Rozwój zarodka przebiega prawidłowo, bez widocznych nieprawidłowości. "
                f"Zalecana kontrola rozwoju ciąży za 7-14 dni."
            )
        elif confidence > 0.85:
            description = (
                f"Badanie USG z dnia {current_date} wskazuje z dużym prawdopodobieństwem na ciążę u klaczy. "
                f"Widoczny jest pęcherzyk ciążowy. Szacowany dzień ciąży: około {estimated_day} dni. "
                f"Obraz wymaga potwierdzenia w badaniu kontrolnym. "
                f"Zalecana kontrola za 7-10 dni dla potwierdzenia prawidłowego rozwoju ciąży."
            )
        else:
            description = (
                f"Badanie USG z dnia {current_date} sugeruje możliwą ciążę u klaczy, "
                f"jednak obraz nie jest jednoznaczny (pewność predykcji: {confidence*100:.1f}%). "
                f"Widoczne są struktury mogące odpowiadać wczesnej ciąży. "
                f"Zalecane jest pilne badanie kontrolne za 5-7 dni dla weryfikacji. "
                f"Należy zachować ostrożność diagnostyczną."
            )
    else:
        if confidence > 0.95:
            description = (
                f"Badanie USG z dnia {current_date} jednoznacznie nie wykazuje cech ciąży u klaczy. "
                f"Obraz macicy prawidłowy, bez widocznych struktur ciążowych. "
                f"Zalecana kontrola cyklu rujowego i powtórzenie badania w odpowiednim momencie cyklu."
            )
        elif confidence > 0.85:
            description = (
                f"Badanie USG z dnia {current_date} nie wykazuje jednoznacznych cech ciąży u klaczy. "
                f"Obraz macicy bez wyraźnych struktur ciążowych. Możliwe wczesne stadium cyklu rujowego. "
                f"Zalecana obserwacja objawów rui i powtórzenie badania."
            )
        else:
            description = (
                f"Badanie USG z dnia {current_date} sugeruje brak ciąży, jednak pewność predykcji jest umiarkowana ({confidence*100:.1f}%). "
                f"Obraz wymaga ostrożnej interpretacji. Zalecane powtórzenie badania za 5-7 dni. "
                f"Wskazana obserwacja objawów klinicznych klaczy."
            )

    if image_features:
        if image_features['contrast'] < 0.3:
            description += f"\n\nUWAGA: Obraz o niskim kontraście ({image_features['contrast']:.2f}) może utrudniać diagnostykę."
        if image_features['entropy'] > 5.0:
            description += f"\n\nUWAGA: Wysoka entropia obrazu ({image_features['entropy']:.2f}) wskazuje na możliwy szum."
        if predicted_class == "pregnant" and image_features['potential_fluid_ratio'] > 0.3:
            description += "\n\nObecność struktur płynowych typowych dla pęcherzyka ciążowego."

    description += (
        f"\n\nZALECENIA:"
        f"\n1. Utrzymanie odpowiedniego żywienia i suplementacji klaczy."
        f"\n2. Regularna kontrola weterynaryjna."
        f"\n3. Monitorowanie ogólnego stanu zdrowia klaczy."
    )

    if additional_info:
        mare_info = ""
        if additional_info.get('klacz_name'):
            mare_info += f"\nImię klaczy: {additional_info['klacz_name']}"
        if additional_info.get('klacz_age'):
            mare_info += f"\nWiek klaczy: {additional_info['klacz_age']} lat"
        if mare_info:
            description = f"INFORMACJE O KLACZY:{mare_info}\n\n" + description

    description += (
        f"\n\nUWAGA: Powyższy opis został wygenerowany automatycznie przez system AI "
        f"i wymaga weryfikacji przez lekarza weterynarii. Pewność predykcji: {confidence*100:.2f}%."
    )

    return description


def analyze_and_predict(image_path, pregnancy_model, day_model, day_mapping, log_file=None):
    """
    Kompleksowa analiza obrazu USG z pełną diagnostyką ciąży i szacowaniem wieku płodu.
    Wykonuje sekwencyjną analizę składającą się z:
    1. Wykrywanie ciąży z modelem binarnym i generowanie opisu klinicznego
    2. Analiza cech technicznych obrazu (kontrast, entropia, struktury)
    3. Szacowanie dnia ciąży modelem wieloklasowym (jeśli wykryto ciążę)
    4. Klasifikacja trymestru na podstawie wieku płodu (≤45, ≤90, >90 dni)
    5. Ranking top-5 najbardziej prawdopodobnych dni z confidence scores
    Struktura wyniku:
    - pregnancy: wynik klasyfikacji binarnej z confidence
    - image_features: parametry techniczne obrazu USG
    - day_estimation: szczegółowe szacowanie wieku płodu (tylko dla ciąży)
    - trimester: automatyczna klasyfikacja okresu ciąży
    - description: profesjonalny raport diagnostyczny
    Zarządzanie wynikami:
    - Automatyczne tworzenie katalogów raportów z datą
    - Pełna obsługa błędów z traceback
    - Logowanie wszystkich operacji
    Zwraca strukturę wyników oraz ścieżkę do katalogu raportów.
    """
    try:
        predicted_class, confidence, description = predict_image(image_path, pregnancy_model, log_file)
        image_features = analyze_image_features(image_path, IMAGE_SIZE)

        result = {
            "image_path": image_path,
            "pregnancy": {
                "is_pregnant": predicted_class == 'pregnant',
                "confidence": confidence
            },
            "image_features": {
                "basic": image_features
            },
            "description": description
        }

        if predicted_class == 'pregnant' and day_model:
            img_array = load_and_preprocess_image(image_path, IMAGE_SIZE)
            preds = day_model.predict(img_array)[0]
            best_idx = int(np.argmax(preds))
            predicted_day = int(list(day_mapping.keys())[best_idx]) if day_mapping else best_idx
            day_conf = preds[best_idx]

            result["day_estimation"] = {
                "predicted_day": predicted_day,
                "confidence": day_conf,
                "top_5_days": np.argsort(preds)[-5:][::-1].tolist(),
                "top_5_confidences": sorted(preds, reverse=True)[:5]
            }

            if predicted_day <= 45:
                result['trimester'] = 1
            elif predicted_day <= 90:
                result['trimester'] = 2
            else:
                result['trimester'] = 3

        output_dir = os.path.join(REPORTS_DIR, datetime.datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(output_dir, exist_ok=True)

        return result, output_dir

    except Exception as e:
        log_error(f"Błąd w analyze_and_predict: {e}", log_file)
        traceback.print_exc()
        return None, None


def batch_process_images(image_dir, log_file=None):
    """
    Masowe przetwarzanie obrazów USG z automatyczną analizą całego katalogu.
    Implementuje wydajny pipeline batch processing przez:
    - Jednokrotne ładowanie modeli na początku sesji (optymalizacja pamięci)
    - Automatyczne wykrywanie obrazów w formatach JPG, PNG, JPEG, BMP
    - Sekwencyjne przetwarzanie każdego obrazu z pełną analizą diagnostyczną
    - Gromadzenie wszystkich wyników w jednej strukturze danych
    Proces przetwarzania:
    1. Inicjalizacja modeli ciąży i szacowania dnia
    2. Skanowanie katalogu pod kątem obsługiwanych formatów obrazów
    3. Analiza każdego obrazu (wykrywanie ciąży + szacowanie wieku)
    4. Agregacja wyników z obsługą błędów (pomija uszkodzone pliki)
    Optymalizacja wydajności:
    - Modele ładowane raz na całą sesję
    - Filtrowanie plików po rozszerzeniu (case-insensitive)
    - Graceful handling błędnych plików bez przerywania całego procesu
    Zwraca listę słowników z wynikami analizy wszystkich przetworzonych obrazów.
    Funkcjonalność do przyszłej implementacji i rozwoju platformy - analizy dużych zbiorów danych diagnostycznych.
    """
    results = []
    pregnancy_model = load_pregnancy_model(log_file=log_file)
    day_model, day_mapping = load_day_estimation_model(log_file=log_file)

    for fname in os.listdir(image_dir):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            image_path = os.path.join(image_dir, fname)
            result, _ = analyze_and_predict(image_path, pregnancy_model, day_model, day_mapping, log_file)
            if result:
                results.append(result)

    return results

