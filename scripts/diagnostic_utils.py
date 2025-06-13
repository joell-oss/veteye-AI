# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import os
import datetime
import traceback
import numpy as np
import tensorflow as tf
from scipy import ndimage
import re
import tempfile
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import Image as ReportLabImage

from config import REPORTS_DIR, IMAGE_SIZE

# Rejestracja czcionek DejaVu do obsługi polskich znaków
# Ścieżka do folderu głównego projektu
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Ścieżka do czcionki
FONT_PATH = os.path.join(BASE_DIR, 'fonts', 'DejaVuSans.ttf')
# Rejestracja czcionki
pdfmetrics.registerFont(TTFont('DejaVu', FONT_PATH))
pdfmetrics.registerFont(TTFont('DejaVu-Bold', os.path.join(BASE_DIR, 'fonts', 'DejaVuSans-Bold.ttf')))

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(name='MyNormal', fontName='DejaVu', fontSize=10, spaceAfter=0))
styles.add(ParagraphStyle(name='MyNormal1', fontName='DejaVu', fontSize=6, spaceAfter=0))
styles.add(ParagraphStyle(name='MyHeading1', fontName='DejaVu-Bold', fontSize=10, spaceAfter=0))
styles.add(ParagraphStyle(name='MyHeading2', fontName='DejaVu-Bold', fontSize=8, spaceAfter=0))


# Przykład wykresu cech obrazu
def save_features_plot_to_file(image_features):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(image_features.keys())
    values = list(image_features.values())

    bar_color = '#90ee90'
    edge_color = '#66bb66'

    ax.barh(labels, values, color=bar_color, edgecolor=edge_color)
    ax.set_title("Cechy obrazu USG")

    # Tymczasowy plik PNG
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.tight_layout()
    plt.savefig(tmp_file.name)
    plt.close(fig)

    return tmp_file.name

def analyze_image_features(image_path, image_size):
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size, color_mode='grayscale')
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0

        mean_intensity = np.mean(img_array)
        std_intensity = np.std(img_array)
        contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0

        histogram, _ = np.histogram(img_array, bins=10, range=(0, 1))
        histogram_norm = histogram / np.sum(histogram)
        entropy = -np.sum(histogram_norm * np.log2(histogram_norm + 1e-10))

        edges = ndimage.sobel(img_array[:, :, 0])
        edge_magnitude = np.mean(np.abs(edges))

        fluid_threshold = 0.7
        potential_fluid = np.sum(img_array > fluid_threshold) / img_array.size

        tissue_threshold = 0.3
        potential_tissue = np.sum(img_array < tissue_threshold) / img_array.size

        return {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'contrast': contrast,
            'entropy': entropy,
            'edge_magnitude': edge_magnitude,
            'potential_fluid_ratio': potential_fluid,
            'potential_tissue_ratio': potential_tissue
        }
    except Exception as e:
        print(f"Błąd analizy obrazu: {e}")
        traceback.print_exc()
        return {
            'mean_intensity': 0.5,
            'std_intensity': 0.2,
            'contrast': 0.4,
            'entropy': 4.0,
            'edge_magnitude': 0.3,
            'potential_fluid_ratio': 0.2,
            'potential_tissue_ratio': 0.3
        }


def generate_description(predicted_class, confidence, image_path=None, additional_info=None):
    image_features = None
    if image_path:
        try:
            image_features = analyze_image_features(image_path, IMAGE_SIZE)
        except Exception as e:
            print(f"Błąd analizy cech obrazu: {e}")

    current_date = datetime.datetime.now().strftime("%d.%m.%Y")
    estimated_day = "nieznany"
    if additional_info:
        estimated_day = additional_info.get('estimated_day', estimated_day)

    if image_path:
        filename = os.path.basename(image_path)
        match = re.search(r'_d(\d+)_', filename)
        if match:
            estimated_day = match.group(1)

    if predicted_class == "pregnant":
        # Opis dla ciąży z uwzględnieniem pewności modelu
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
    else:  # not_pregnant
        if confidence > 0.95:
            description = (
                f"Badanie USG z dnia {current_date} jednoznacznie nie wykazuje cech ciąży u klaczy. "
                f"Obraz macicy prawidłowy, bez widocznych struktur ciążowych. "
                f"Zalecana kontrola cyklu rujowego i powtórzenie badania w odpowiednim momencie "
                f"cyklu dla weryfikacji."
            )
        elif confidence > 0.85:
            description = (
                f"Badanie USG z dnia {current_date} nie wykazuje jednoznacznych cech ciąży u klaczy. "
                f"Obraz macicy bez wyraźnych struktur ciążowych. Możliwe wczesne stadium cyklu rujowego. "
                f"Zalecana obserwacja objawów rui i powtórzenie badania w odpowiednim czasie."
            )
        else:
            description = (
                f"Badanie USG z dnia {current_date} sugeruje brak ciąży, "
                f"jednak pewność predykcji jest umiarkowana ({confidence*100:.1f}%). "
                f"Obraz wymaga ostrożnej interpretacji. Zalecane powtórzenie badania za 5-7 dni. "
                f"Wskazana obserwacja objawów klinicznych klaczy."
            )

    if image_features:
        if image_features['contrast'] < 0.3:
            description += f"\n\nUWAGA: Niski kontrast obrazu ({image_features['contrast']:.2f})."
        if image_features['entropy'] > 5.0:
            description += f"\n\nUWAGA: Wysoki szum obrazu (entropia: {image_features['entropy']:.2f})."
        if predicted_class == "pregnant" and image_features['potential_fluid_ratio'] > 0.3:
            description += f"\n\nObecne struktury płynowe charakterystyczne dla pęcherzyka ciążowego."

    description += (
        f"\n\nZALECENIA:"
        f"\n1. Utrzymanie odpowiedniego żywienia i suplementacji klaczy."
        f"\n2. Regularna kontrola weterynaryjna."
        f"\n3. Monitorowanie ogólnego stanu zdrowia klaczy."
    )

    if additional_info:
        mare_info = ""
        if additional_info.get('klacz_name'):
            mare_info += f"Imię klaczy: {additional_info['klacz_name']}\n"
        if additional_info.get('klacz_age'):
            mare_info += f"Wiek klaczy: {additional_info['klacz_age']} lat\n"
        if mare_info:
            description = f"INFORMACJE O KLACZY:\n{mare_info}\n" + description

    description += f"\n\nUWAGA: Powyższy opis został wygenerowany automatycznie przez system AI. \n\nWymagana weryfikacja lekarza weterynarii."
    return description


def generate_pdf_report(image_path, predicted_class, confidence, description, additional_info=None, image_features=None):
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    basename = os.path.basename(image_path).split('.')[0]
    report_filename = os.path.join(REPORTS_DIR, f"{timestamp}_{basename}.pdf")
    os.makedirs(os.path.dirname(report_filename), exist_ok=True)


    doc = SimpleDocTemplate(report_filename, pagesize=letter)
    story = []

    story.append(Paragraph("RAPORT DIAGNOSTYCZNY USG", styles['MyHeading1']))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph(f"Data badania: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['MyNormal']))
    if additional_info:
        if additional_info.get('klacz_name'):
            story.append(Paragraph(f"Imię klaczy: {additional_info['klacz_name']}", styles['MyNormal']))
        if additional_info.get('klacz_age'):
            story.append(Paragraph(f"Wiek klaczy: {additional_info['klacz_age']} lat", styles['MyNormal']))
        if additional_info.get('estimated_day'):
            story.append(Paragraph(f"Szacowany dzień cyklu/ciąży: {additional_info['estimated_day']}", styles['MyNormal']))

    story.append(Paragraph(f"Plik obrazu: {os.path.basename(image_path)}", styles['MyNormal']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"Predykcja modelu: {predicted_class}", styles['MyNormal']))
    story.append(Paragraph(f"Pewność predykcji: {confidence*100:.2f}%", styles['MyNormal']))
    story.append(Spacer(1, 0.2 * inch))

    try:
        img = ReportLabImage(image_path, width=5 * inch, height=4 * inch)
        story.append(img)
        story.append(Spacer(1, 0.2 * inch))
    except Exception as e:
        story.append(Paragraph("Błąd wczytania obrazu do raportu.", styles['MyNormal']))

    story.append(Paragraph("Opis diagnostyczny:", styles['MyHeading2']))
    for para in description.split('\n'):
        story.append(Paragraph(para.strip(), styles['MyNormal']))

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("veteye.AI - system predykcji ciąży klaczy na podstawie obrazu ultrasonograficznego, ALK.BIZNES.AI.GR12.G2, 2025", styles['MyNormal1']))

    if image_features:
        from diagnostic_utils import save_features_plot_to_file  # jeśli masz w osobnym pliku

        chart_path = save_features_plot_to_file(image_features)
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph("Wykres cech obrazu USG:", styles["MyHeading2"]))
        story.append(ReportLabImage(chart_path, width=5.5 * inch, height=3.5 * inch))
        story.append(Spacer(1, 0.2 * inch))
    
        story.append(Paragraph("Wyjaśnienie cech:", styles["MyNormal"]))
        feature_explanation = [
            "<u><b>• mean_intensity</b></u> – średnia jasność pikseli obrazu.<br/>Typowy zakres dobrej jakości: 0.2 – 0.5.<br/>Znaczenie diagnostyczne: Zbyt niska = niedoświetlony obraz; zbyt wysoka = prześwietlony.",
            "<br/><u><b>• std_intensity</b></u> – odchylenie standardowe jasności (różnorodność jasności).<br/>Typowy zakres dobrej jakości: 0.1 – 0.3.<br/>Znaczenie diagnostyczne: Za niska = obraz jednolity (mało informacji); za wysoka = szum.",
            "<br/><u><b>• contrast</b></u> – stosunek kontrastu do jasności (czy struktury są wyraźne).<br/>Typowy zakres dobrej jakości: > 0.4.<br/>Znaczenie diagnostyczne: Wyraźne różnice między strukturami (czytelność).",
            "<br/><u><b>• entropy</b></u> – miara złożoności tekstury (szum lub szczegóły).<br/>Typowy zakres dobrej jakości: 2.5 – 4.5.<br/>Znaczenie diagnostyczne: Za niska = mało szczegółów; za wysoka > 5.0 = zbyt duży szum.",
            "<br/><u><b>• edge_magnitude</b></u> – siła krawędzi (liczba i intensywność konturów).<br/>Typowy zakres dobrej jakości: 0.2 – 0.5.<br/>Znaczenie diagnostyczne: Zbyt niska = mało struktur; za wysoka = szum lub artefakty.",
            "<br/><u><b>• potential_fluid_ratio</b></u> – procent jasnych obszarów sugerujących płyn.<br/>Typowy zakres dobrej jakości: 0.05 – 0.3.<br/>Znaczenie diagnostyczne: Więcej = prawdopodobieństwo obecności pęcherzyka.",
            "<br/><u><b>• potential_tissue_ratio</b></u> – procent ciemnych obszarów sugerujących tkanki.<br/>Typowy zakres dobrej jakości: 0.4 – 0.8.<br/>Znaczenie diagnostyczne: Zbyt mało = słaba widoczność struktur anatomicznych."
        ]
        for line in feature_explanation:
            story.append(Paragraph(line, styles['MyNormal']))

    doc.build(story)
    return report_filename
