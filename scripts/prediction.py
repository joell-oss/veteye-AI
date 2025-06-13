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
    """Ładuje model wykrywania ciąży"""
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
    """Ładuje model szacowania dnia ciąży oraz mapowanie dni"""
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
    """Predykcja ciąży na podstawie przygotowanego obrazu"""
    prediction = model.predict(img_array)[0]
    confidence = float(prediction[0]) if prediction.shape[0] == 1 else float(prediction)
    return 'pregnant' if confidence > 0.5 else 'not_pregnant', confidence


def estimate_pregnancy_day(img_array, model, mapping=None):
    """Szacowanie dnia ciąży i top 5 klas"""
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
    """Przetwarza obraz i generuje wynik oraz opis"""
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

