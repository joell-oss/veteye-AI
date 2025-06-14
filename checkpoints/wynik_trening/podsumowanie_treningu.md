# Podsumowanie treningu modelu: Klasyfikacja obraz�w USG klaczy (ci��a vs brak ci��y)

## 1. Architektura modelu

![Architektura modelu](architektura_modelu_v1.png)

Model oparty o architektur� InceptionV3, dostosowany do klasyfikacji obraz�w USG na dwie klasy: `pregnant` i `not_pregnant`.
Model bazuje na architekturze InceptionV3 z wej�ciem o rozmiarze 380x380 pikseli. Na szczycie modelu znajduje si� w�asna, 
dopasowana g�owica klasyfikacyjna. Takie rozwi�zanie zapewnia wysok� skuteczno�� oraz zdolno�� do generalizacji 
nawet przy ograniczonych zbiorach danych medycznych.
---

## 2. Przebieg treningu

- **Rozmiar wej�ciowy:** 380x380 pikseli  
- **Zbi�r treningowy:** 900 obraz�w  
- **Zbi�r testowy:** 218 obraz�w  
- **Epoki:** 60 (bazowy) + 40 (fine-tuning)  
- **Batch size:** 16  
- **Klasy:** `not_pregnant`, `pregnant`

---

### Historia treningu � model bazowy

#### Dok�adno��

![Dok�adno�� � model bazowy](training_history_base.png)

**Opis:**  
Szybki wzrost dok�adno�ci, stabilizacja na poziomie powy�ej 96% ju� po kilku epokach - systematyczny wzrost dok�adno�ci 
zar�wno dla zbioru treningowego, jak i walidacyjnego. Od pocz�tku walidacyjna accuracy przekracza 0.96 i zbli�a si� do 
warto�ci 0.99 ju� po kilku epokach, co potwierdza szybkie uczenie si� modelu i brak powa�nego overfittingu.

#### Krzywa AUC

![AUC � model bazowy](training_history_base_auc.png)

**Opis:**  
AUC zbli�one do 1, bardzo dobra rozdzielczo�� klas ju� od pocz�tku treningu. Warto�� AUC (Area Under Curve) jest bardzo 
wysoka przez ca�y proces treningu. �wiadczy to o zdolno�ci modelu do poprawnej klasyfikacji zar�wno pozytywnych, 
jak i negatywnych przypadk�w nawet przy r�nych progach decyzyjnych.

#### Precision & Recall

![Precision & Recall � model bazowy](training_history_base_precision_recall.png)

**Opis:**  
Precyzja i recall na bardzo wysokim poziomie przez ca�y trening, bez wyra�nego overfittingu. Zar�wno precyzja, 
jak i recall utrzymuj� si� na wysokim, stabilnym poziomie, z minimalnymi r�nicami pomi�dzy zbiorem treningowym 
a walidacyjnym, co oznacza, �e model dobrze generalizuje.

---

### Fine-tuning

#### Dok�adno��

![Dok�adno�� � fine-tuning](training_history_finetuned.png)

**Opis:**
Podczas etapu fine-tuningu widoczna jest dalsza poprawa wynik�w oraz utrzymanie bardzo wysokiej dok�adno�ci na 
zbiorze walidacyjnym. Ostatecznie model osi�ga praktycznie maksymaln� skuteczno�� klasyfikacji.

#### Krzywa AUC

![AUC � fine-tuning](training_history_finetuned_auc.png)

**Opis:**
Warto�� AUC pozostaje bliska 1, co oznacza, �e model nie tylko trafnie klasyfikuje, ale te� bardzo dobrze 
klasyfikuje poprawnie z du�� pewno�ci� predykcji.

#### Precision & Recall

![Precision & Recall � fine-tuning](training_history_finetuned_precision_recall.png)

**Opis:**  
Model po fine-tuningu utrzymuje i lekko poprawia wysokie warto�ci dok�adno�ci, AUC oraz precyzji i recall. 
Precision, jak i recall osi�gaj� bardzo wysokie warto�ci (powy�ej 0.97) dla obu klas, nawet po fine-tuningu, 
co �wiadczy o stabilno�ci i skuteczno�ci modelu.

---

## 3. Wyniki ko�cowe

### Raport klasyfikacji (zbi�r testowy)

| Klasa         | Precision | Recall | F1-score | Support |
|---------------|-----------|--------|----------|---------|
| not_pregnant  | 1.00      | 0.97   | 0.99     | 105     |
| pregnant      | 0.97      | 1.00   | 0.99     | 113     |
| **accuracy**  |           |        | **0.99** | 218     |
| macro avg     | 0.99      | 0.99   | 0.99     | 218     |
| weighted avg  | 0.99      | 0.99   | 0.99     | 218     |

---

Model osi�gn�� bardzo wysok� skuteczno�� (accuracy 0.99) z r�wnowag� pomi�dzy precyzj� a recall dla obu klas. 
Praktycznie brak b��dnych klasyfikacji.

### Macierz pomy�ek

![Macierz pomy�ek](confusion_matrix.png)

**Opis:**  
Model poprawnie klasyfikuje zdecydowan� wi�kszo�� przypadk�w. Zaledwie kilka pr�bek zosta�o b��dnie sklasyfikowanych 
� co w praktyce oznacza bardzo wysok� niezawodno�� rozwi�zania.

---

### Krzywa ROC

![Krzywa ROC](roc_curve.png)

**Opis:**  
Krzywa ROC znajduje si� bardzo blisko lewego g�rnego rogu wykresu, co wskazuje na niemal perfekcyjn� skuteczno�� modelu. 
Wysoka warto�� AUC (bliska 1.0) potwierdza doskona�� rozdzielczo�� klas.

---

### Precision-Recall

![Precision-Recall Curve](precision_recall_curve.png)

**Opis:**  
Krzywa precision-recall utrzymuje si� bardzo wysoko dla obu klas, co oznacza, �e model nie tylko przewiduje poprawnie, 
ale te� jest odporny na fa�szywie pozytywne i negatywne wskazania.

---

## 4. Podsumowanie i rekomendacje

- **Accuracy:** 0.99  
- **Precision/Recall/F1:** >0.97 dla obu klas  
- **Brak overfittingu**, bardzo dobra generalizacja  
- **Model gotowy do walidacji na nowych zbiorach i wdro�enia produkcyjnego**

---

| Wnioski                                                                                                          |
|------------------------------------------------------------------------------------------------------------------|
| Model osi�ga 99% accuracy oraz bardzo wysokie warto�ci precision, recall i f1-score.                             |
| Wysokie warto�ci AUC oraz znakomita macierz pomy�ek wskazuj� na niezawodno�� klasyfikatora.                      |
| Zar�wno w fazie treningu bazowego, jak i fine-tuningu model stabilnie si� uczy� i nie przejawia� oznak overfittingu. |
| Model jest gotowy do wdro�enia lub dalszych test�w na nowych zbiorach danych.                                    |


| Rekomendacje                                                                                                 |
|-------------------------------------------------------------------------------------------------------------|
| Dla pe�nej walidacji warto przetestowa� model na zupe�nie nowych, niezale�nych danych medycznych.           |
| Sugerowane jest dalsze monitorowanie wynik�w modelu po wdro�eniu.                                           |



**Wszystkie wykresy i grafiki s� bezpo�rednio generowane w procesie treningu i odzwierciedlaj� stabilno�� oraz wysok� jako�� modelu.**


