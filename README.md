
# System wykrywania ciąży u klaczy - projekt edukacyjny ALK.BIZNES.AI.GR12G2

System służy do automatycznego wykrywania ciąży oraz szacowania dnia ciąży na podstawie obrazów USG. Projekt wspiera zarówno analizę pojedynczych obrazów, jak i przetwarzanie wsadowe z wykorzystaniem GUI i modeli uczenia maszynowego.

---

## Struktura projektu

| Moduł                                | Pliki                                                                                                                                                                    |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Interfejs graficzny (GUI)**        | `analysis_gui.py`, `gui_utils.py`, `run_analysis_gui.py`, `web_gui.py`                                                                                                   |
| **Analiza obrazów i diagnostyka**    | `image_analysis.py`, `diagnostic_utils.py`                                                                                                                               |
| **Pobieranie danych**                | `download_files.py`, `download_files_notP.py`, `download_files_YT.py`                                                                                                    |
| **Trenowanie modeli**                | `model_training.py`, `train_day_estimation_model.py`, `train_pregnancy_model.py`, `podzial_training_test.py`                                                             |
| **Przygotowanie danych**             | `prepare_dataset.py`                                                                                                                                                     |
| **Raportowanie**                     | `report_generator.py`                                                                                                                                                    |
| **Inne skrypty**                     | `augmentacja_not_pragnent.py`, `batch_processing.py`, `data_loader.py`, `evaluation.py`, `main.py`, `model_builder.py`, `prediction.py`, `pregnancy_detection_script.py` |
| **Narzędzia i konfiguracja**         | `config.py`, `logging_utils.py`                                                                                                                                          |
| **Uruchamianie i pliki wykonywalne** | `#StartMENU.ps1`, `yt-dlp.exe`                                                                                                                                           |
| **Dokumentacja**                     | `README.md`                                                                                                                                                              |

---

## Funkcjonalności

- **Wykrywanie ciąży** – klasyfikacja obrazu jako „ciąża” lub „brak ciąży”.
- **Szacowanie dnia ciąży** – regresja liczby dni od zapłodnienia.
- **Analiza obrazów USG** – automatyczne wykrywanie cech diagnostycznych.
- **Raportowanie PDF** – generowanie raportów diagnostycznych z wynikami analizy.
- **GUI i Web GUI** – graficzne interfejsy do ręcznej analizy i przetwarzania wsadowego.
- **Trenowanie modeli** – trenowanie własnych modeli klasyfikacyjnych i regresyjnych.
- **Pobieranie danych z YouTube** – automatyczne pobieranie nagrań do analizy.

---

## Wymagania

- Python 3.10
- Biblioteki:
  - `TensorFlow >= 2.8`
  - `OpenCV >= 4.5`
  - `NumPy`, `Matplotlib`, `Scikit-image`, `Pandas`
  - `ReportLab` – do generowania raportów PDF
  - `Tkinter` – GUI desktopowe
  - `Flask` lub `FastAPI` – GUI webowe (dla `web_gui.py`)
- `yt-dlp` – do pobierania wideo z YouTube

## Zewnętrzne komponenty

Projekt korzysta z narzędzia [yt-dlp](https://github.com/yt-dlp/yt-dlp), dostępnego na licencji [Unlicense](https://github.com/yt-dlp/yt-dlp/blob/master/LICENSE).

W katalogu `scripts/` znajduje się skompilowana wersja `yt-dlp.exe`, pobrana z oficjalnego źródła.

---

# Obsługa GPU (Windows) – instalacja cuDNN 8.1 (CUDA 11.2)

Aby korzystać z akceleracji GPU, **wymagane jest środowisko CUDA 11.2 oraz biblioteka cuDNN v8.1.1**.  
Opis pobrania oraz instalacji.

> **Plik ZIP należy zapisać w katalogu projektu `package/`.**

## Krok 1 · Pobranie

1. Wejdź na stronę [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) ↗ i zaloguj się (konto NVIDIA Developer jest bezpłatne).
2. Pobierz wersję: **cuDNN v8.1.1 for CUDA 11.2 (Windows x64)** – plik **`cudnn-11.2-windows-x64-v8.1.1.33.zip`**.
3. Skopiuj pobrane archiwum do folderu projektu:
   ```text
   package/cudnn-11.2-windows-x64-v8.1.1.33.zip
   ```

## Krok 2 · Instalacja

1. **Rozpakuj archiwum:**
   ```powershell
   Expand-Archive package\cudnn-11.2-windows-x64-v8.1.1.33.zip -DestinationPath .
   ```

2. **Skopiuj zawartość podkatalogów** do odpowiadających ścieżek instalacji CUDA 11.2 (domyślnie `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`):

   | Źródło w archiwum | Cel |
   |-------------------|-----|
   | `bin\*` | `...\CUDA\v11.2\bin` |
   | `include\*` | `...\CUDA\v11.2\include` |
   | `lib\x64\*` | `...\CUDA\v11.2\lib\x64` |

3. **Upewnij się**, że w `bin` znajdują się pliki `cudnn64_8.dll`, `cudnn_ops_infer64_8.dll`, `cudnn_cnn_infer64_8.dll`, itp.

## Krok 3 · Zmienne środowiskowe

Ścieżki CUDA powinny znajdować się w PATH (instalator CUDA dodaje je automatycznie).

**W razie potrzeby:**
```powershell
$env:Path += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin"
$env:Path += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64"
```

## Krok 4 · Weryfikacja

```powershell
python - << "EOF"
import tensorflow as tf
print("TF-GPU:", tf.test.is_built_with_cuda())
print("GPU dostępne:", tf.config.list_physical_devices('GPU'))
EOF
```

Jeśli oba polecenia zwracają `True` lub listę urządzeń, instalacja zakończyła się sukcesem.

---

**Uwaga:** cuDNN jest udostępniany na licencji NVIDIA Software License Agreement.  
Korzystanie wymaga akceptacji warunków tej licencji.

---

## Instalacja

```bash
Utwórz i aktywuj wirtualne środowisko:
python -m venv venv
source venv/bin/activate  # Na Windows: venv\Scripts\activate
W środowisku conda:
conda create --prefix D:/python/veteye2/veteye2_env python=3.10

Aktywacja środowiska:
conda activate D:\python\veteye2\eteye2_env

pip install -r requirements.txt
```

---

## Struktura katalogów

Aby poprawnie korzystać z systemu, należy utworzyć następującą strukturę katalogów:
```
USG-Mares-Pregnancy-Dataset/
├── Training/              # Dane treningowe
│   ├── pregnant/          # Obrazy USG ciąży
│   └── not_pregnant/      # Obrazy USG bez ciąży
└── Test/                  # Dane testowe
├── pregnant/          # Obrazy USG ciąży do testu
└── not_pregnant/      # Obrazy USG bez ciąży do testu
```
Dla modelu szacowania dnia ciąży:
```text
USG-Mares-Pregnancy-Days/
├── 20/                   # Obrazy USG dla ciąży 20-dniowej
├── 30/                   # Obrazy USG dla ciąży 30-dniowej
├── 45/                   # Obrazy USG dla ciąży 45-dniowej
└── ...                   # Inne dni ciąży
```
### Przygotowanie danych

Aby przygotować dane z istniejącego katalogu obrazów:
python main.py --prepare --source-dir /sciezka/do/obrazow --structure binary --augment

Opcje:
- `--source-dir`: Ścieżka do katalogu z obrazami USG
- `--dest-dir`: Katalog docelowy na zestaw danych (domyślnie: USG-Mares-Pregnancy-Dataset)
- `--structure`: Struktura danych: 'binary' dla ciąża/brak ciąży, 'days' dla dni ciąży
- `--augment`: Zastosuj augmentację danych

### Trenowanie modelu wykrywania ciąży
python main.py --train --model-type pregnancy --train-dir USG-Mares-Pregnancy-Dataset

Opcje:
- `--train-dir`: Katalog z danymi treningowymi
- `--resume`: Wznów trening od ostatniego punktu kontrolnego

### Trenowanie modelu szacowania dnia ciąży
python main.py --train --model-type day --train-dir USG-Mares-Pregnancy-Days

## Podsumowanie treningu
[Zobacz szczegóły](checkpoints/wynik_trening/podsumowanie_treningu.md)

## Uruchamianie GUI

![Zrzut ekranu GUI](logs/gui.png)

Opis: Główne okno aplikacji desktopowej.

python main.py --analyze

Opcje:
- `--model`: Ścieżka do modelu do używania (domyślnie: najnowszy model)

### Analiza pojedynczego obrazu
python main.py --analyze --image /sciezka/do/obrazu.jpg

### Przetwarzanie wsadowe
python main.py --batch --input-dir /sciezka/do/katalogu/z/obrazami --report

Opcje:
- `--input-dir`: Katalog wejściowy z obrazami USG
- `--output-dir`: Katalog wyjściowy na wyniki
- `--report`: Generuj raport zbiorczy PDF

**Web GUI:**
```bash
python web_gui.py
```
![Zrzut ekranu WebGUI](logs/webgui.png)

Opis: Widok panelu użytkownika w aplikacji webowej.

---

## Opis modułów

- `config.py` - Konfiguracja systemu
- `logging_utils.py` - Narzędzia do logowania
- `data_loader.py` - Ładowanie i przetwarzanie danych
- `model_builder.py` - Budowa modeli
- `model_training.py` - Trening modeli
- `evaluation.py` - Ewaluacja modeli
- `image_analysis.py` - Analiza cech obrazów USG
- `prediction.py` - Predykcje z wykorzystaniem modeli
- `report_generator.py` - Generowanie raportów diagnostycznych
- `gui_utils.py` - Narzędzia dla interfejsu graficznego
- `analysis_gui.py` - Główny interfejs analizy obrazów USG
- `batch_processing.py` - Skrypt do przetwarzania wsadowego
- `train_pregnancy_model.py` - Skrypt do treningu modelu wykrywania ciąży
- `train_day_estimation_model.py` - Skrypt do treningu modelu szacowania dnia ciąży
- `run_analysis_gui.py` - Skrypt uruchamiający interfejs graficzny
- `prepare_dataset.py` - Skrypt do przygotowania zestawu danych
- `main.py` - Główny skrypt uruchamiający system

## Opis interfejsu graficznego

### Główne okno

Interfejs graficzny zawiera następujące elementy:
- Panel obrazu USG - wyświetla aktualnie analizowany obraz
- Panel wyników - wyświetla wyniki analizy i cechy obrazu
- Przyciski akcji - wczytywanie, analizowanie, generowanie raportów

### Funkcje interfejsu

1. **Wczytaj obraz** - Pozwala wybrać obraz USG do analizy
2. **Analizuj obraz** - Przeprowadza analizę wczytanego obrazu
3. **Generuj raport** - Tworzy raport PDF z wynikami analizy
4. **Analiza wsadowa** - Uruchamia przetwarzanie wielu obrazów
5. **Ustawienia** - Umożliwia konfigurację parametrów systemu

## Raporty diagnostyczne

System generuje raporty PDF z wynikami analizy, które zawierają:
- Informacje o ciąży (wykryta/nie wykryta)
- Szacowany dzień ciąży (jeśli wykryto ciążę)
- Analizę cech obrazu USG
- Wizualizację analizy
- Przybliżone daty krycia i porodu

## Licencja

Ten projekt jest dostępny na licencji: Proprietary / Internal Evaluation Use Only. Szczegóły w pliku LICENSE.

## Autorzy
KOZMINSKI EXECUTIVE BUSINESS SCHOOL
CENTRUM DORADZTWA I KSZTAŁCENIA MENEDŻERÓW 
Kierunek studiów: Biznes.ai: zarządzanie projektami sztucznej inteligencji, edycja 12
Projekt końcowy: Wdrożenie platformy wsparcia weterynaryjnej diagnostyki USG z wykorzystaniem AI
Jan Polański, 			nr albumu 67192-CKP
Paweł Rusek, 			nr albumu 67193-CKP
Józef Sroka, 			nr albumu 67195-CKP
Krzysztof Trawiński, 	nr albumu 67201-CKP
Iwona Grub-Malinowska, 	nr albumu 67409-CKP
Adam Lasko, 			nr albumu 67182-CKP

## Trening modeli

- Klasyfikator ciąży: `train_pregnancy_model.py`
- Model szacowania dnia (planowany w wersji produkcyjnej): `train_day_estimation_model.py`
- Przygotowanie danych: `prepare_dataset.py`

---

## Generowanie raportów

Raporty PDF tworzone są automatycznie po analizie obrazu i zawierają:
- Obraz źródłowy
- Wynik klasyfikacji
- Szacowany dzień ciąży
- Cechy diagnostyczne

![Raporty generowane w systemie](logs/colageusg.jpg)

Opis: Raporty PDF z analizy obrazu USG.

# Uruchomienie demonstartora

![Panel demonstartora](logs/panel.png)

Opis: Panel demonstratora - menu.

1. Otwórz PowerShell w katalogu projektu.
2. Uruchom skrypt komendą:
    ```powershell
    .\#StartMENU.ps1
    ```
3. Wybierz odpowiednią opcję wpisując jej numer i zatwierdź Enter.

## Menu Główne – Opcje

1. **Pełny Proces**  
   Uruchamia kompletny cykl:
   - Trenowanie modelu wykrywania ciąży
   - Opcjonalne trenowanie modelu szacowania dni ciąży
   - Uruchomienie interfejsu graficznego do analizy

2. **Tylko Analiza**  
   Wykorzystuje gotowe, wytrenowane modele do analizy:
   - Interfejs graficzny
   - Analiza pojedynczego obrazu
   - Przetwarzanie wsadowe

3. **Interfejs Graficzny**  
   Uruchamia aplikację z graficznym interfejsem użytkownika dla interaktywnej analizy obrazów.

4. **Analiza Pojedynczego Obrazu**  
   Szybka analiza wybranego pliku obrazu USG z wyświetleniem wyników diagnostyki.

5. **Przetwarzanie Wsadowe**  
   Masowe przetwarzanie wielu obrazów z opcją generowania raportu zbiorczego.

6. **Ewaluacja Modelu**  
   Testowanie i ocena skuteczności istniejących modeli na danych testowych.

7. **Wznowienie Treningu**  
   Kontynuacja treningu modelu z wcześniej zapisanego punktu kontrolnego.

8. **Interfejs Webowy**  
   Uruchomienie serwera webowego z interfejsem dostępnym przez przeglądarkę.
