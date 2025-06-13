# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import os
import numpy as np
import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from logging_utils import log_info, log_error, log_section, log_success
from config import REPORTS_DIR

def evaluate_pregnancy_model(model, test_generator, class_names, output_dir=None, log_file=None):
    """Ocenia model wykrywania ciąży i generuje raport ewaluacyjny"""
    
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(REPORTS_DIR, f"pregnancy_evaluation_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    log_section("Ewaluacja modelu wykrywania ciąży", log_file)
    
    try:
        # Reset generatora
        test_generator.reset()
        
        # Wykonaj predykcje
        log_info("Wykonywanie predykcji na zbiorze testowym...", log_file)
        predictions = model.predict(test_generator, verbose=1)
        
        # Konwersja predykcji na klasy
        y_pred = (predictions > 0.5).astype(int).flatten()
        y_true = test_generator.classes
        
        # Raport klasyfikacji
        log_info("\nRaport klasyfikacji:", log_file)
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        report_text = classification_report(y_true, y_pred, target_names=class_names)
        log_info("\n" + report_text, log_file)
        
        # Zapisz raport do pliku tekstowego
        report_file = os.path.join(output_dir, "classification_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"Data ewaluacji: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(report_text)
        
        # Zapisz metryki do pliku CSV
        metrics_file = os.path.join(output_dir, "metrics.csv")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write("Class,Precision,Recall,F1-Score,Support\n")
            for cls in class_names:
                f.write(f"{cls},{report[cls]['precision']:.4f},{report[cls]['recall']:.4f},{report[cls]['f1-score']:.4f},{report[cls]['support']}\n")
            f.write(f"accuracy,,,,{report['accuracy']:.4f}\n")
            f.write(f"macro avg,{report['macro avg']['precision']:.4f},{report['macro avg']['recall']:.4f},{report['macro avg']['f1-score']:.4f},{report['macro avg']['support']}\n")
            f.write(f"weighted avg,{report['weighted avg']['precision']:.4f},{report['weighted avg']['recall']:.4f},{report['weighted avg']['f1-score']:.4f},{report['weighted avg']['support']}\n")
        
        # Macierz pomyłek
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Macierz pomyłek')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Dodanie wartości do macierzy
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('Prawdziwa etykieta')
        plt.xlabel('Przewidziana etykieta')
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
        
        # Zapisz wyniki jako JSON
        results = {
            "accuracy": float(report['accuracy']),
            "precision": float(report['weighted avg']['precision']),
            "recall": float(report['weighted avg']['recall']),
            "f1": float(report['weighted avg']['f1-score']),
            "class_metrics": {cls: {k: float(v) for k, v in metrics.items() if k != 'support'} 
                              for cls, metrics in report.items() if cls in class_names}
        }
        
        results_file = os.path.join(output_dir, "evaluation_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
        log_success(f"Ewaluacja zakończona. Raporty zapisane w {output_dir}", log_file)
        
        return results
    
    except Exception as e:
        log_error("Błąd podczas ewaluacji modelu wykrywania ciąży", e, log_file)
        return None

def evaluate_day_estimation_model(model, test_generator, day_mapping, output_dir=None, log_file=None):
    """Ocenia model szacowania dnia ciąży i generuje raport ewaluacyjny"""
    
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(REPORTS_DIR, f"day_estimation_evaluation_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    log_section("Ewaluacja modelu szacowania dnia ciąży", log_file)
    
    try:
        # Reset generatora
        test_generator.reset()
        
        # Wykonaj predykcje
        log_info("Wykonywanie predykcji na zbiorze testowym...", log_file)
        predictions = model.predict(test_generator, verbose=1)
        
        # Pobierz prawdziwe etykiety
        y_true = test_generator.classes
        
        # Oblicz predykcje klas
        y_pred = np.argmax(predictions, axis=1)
        
        # Mapuj indeksy na dni ciąży
        true_days = np.array([day_mapping.get(str(idx), idx) for idx in y_true])
        pred_days = np.array([day_mapping.get(str(idx), idx) for idx in y_pred])
        
        # Oblicz różnice w dniach
        day_differences = np.abs(true_days - pred_days)
        
        # Oblicz statystyki błędów
        mean_absolute_error = np.mean(day_differences)
        median_absolute_error = np.median(day_differences)
        accuracy_within_7_days = np.mean(day_differences <= 7) * 100
        accuracy_within_14_days = np.mean(day_differences <= 14) * 100
        accuracy_within_30_days = np.mean(day_differences <= 30) * 100
        
        # Histogram różnic dni
        plt.figure(figsize=(12, 8))
        plt.hist(day_differences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(mean_absolute_error, color='red', linestyle='dashed', linewidth=2, label=f'Średni błąd: {mean_absolute_error:.1f} dni')
        plt.axvline(median_absolute_error, color='green', linestyle='dashed', linewidth=2, label=f'Mediana błędu: {median_absolute_error:.1f} dni')
        plt.title('Rozkład błędów szacowania dnia ciąży')
        plt.xlabel('Błąd bezwzględny (dni)')
        plt.ylabel('Liczba przypadków')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "day_error_histogram.png"))
        plt.close()
        
        # Zapisz raport do pliku tekstowego
        report_file = os.path.join(output_dir, "day_estimation_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"Raport ewaluacji modelu szacowania dnia ciąży\n")
            f.write(f"Data: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Średni błąd bezwzględny: {mean_absolute_error:.2f} dni\n")
            f.write(f"Mediana błędu bezwzględnego: {median_absolute_error:.2f} dni\n")
            f.write(f"Dokładność w zakresie 7 dni: {accuracy_within_7_days:.2f}%\n")
            f.write(f"Dokładność w zakresie 14 dni: {accuracy_within_14_days:.2f}%\n")
            f.write(f"Dokładność w zakresie 30 dni: {accuracy_within_30_days:.2f}%\n")
        
        # Zapisz szczegółowe wyniki do CSV
        results_df = pd.DataFrame({
            'True_Day': true_days,
            'Predicted_Day': pred_days,
            'Absolute_Error': day_differences
        })
        results_df.to_csv(os.path.join(output_dir, "day_estimation_results.csv"), index=False)
        
        # Zapisz statystyki jako JSON
        stats = {
            "mean_absolute_error": float(mean_absolute_error),
            "median_absolute_error": float(median_absolute_error),
            "accuracy_within_7_days": float(accuracy_within_7_days),
            "accuracy_within_14_days": float(accuracy_within_14_days),
            "accuracy_within_30_days": float(accuracy_within_30_days)
        }
        
        stats_file = os.path.join(output_dir, "day_estimation_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4)
        
        # Wykres rozrzutu prawdziwe vs przewidywane dni
        plt.figure(figsize=(10, 10))
        plt.scatter(true_days, pred_days, alpha=0.5)
        plt.plot([min(true_days), max(true_days)], [min(true_days), max(true_days)], 'r--')
        plt.xlabel('Prawdziwy dzień ciąży')
        plt.ylabel('Przewidywany dzień ciąży')
        plt.title('Prawdziwe vs Przewidywane dni ciąży')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "true_vs_predicted_days.png"))
        plt.close()
        
        log_success(f"Ewaluacja modelu szacowania dnia zakończona. Raporty zapisane w {output_dir}", log_file)
        
        return stats
    
    except Exception as e:
        log_error("Błąd podczas ewaluacji modelu szacowania dnia ciąży", e, log_file)
        return None
