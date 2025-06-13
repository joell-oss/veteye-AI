# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from logging_utils import log_info, log_error, log_success
from config import REPORTS_DIR, MIN_PREGNANCY_DAY, MAX_PREGNANCY_DAY


def create_pregnancy_report(analysis_result, output_dir=None, include_images=True, log_file=None):
    """Generuje raport PDF z wynikami analizy obrazu USG klaczy"""
    
    if output_dir is None:
        output_dir = os.path.dirname(analysis_result.get("image_path", ""))
        if not output_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(REPORTS_DIR, f"report_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Nazwa pliku raportu
    report_filename = os.path.join(output_dir, "raport_diagnostyczny.pdf")
    
    try:
        # Tworzenie dokumentu PDF
        doc = SimpleDocTemplate(
            report_filename,
            pagesize=A4,
            rightMargin=1.5*cm,
            leftMargin=1.5*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        # Style
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='Title',
            parent=styles['Heading1'],
            fontSize=18,
            alignment=1,  # Wyśrodkowany
            spaceAfter=12
        ))
        styles.add(ParagraphStyle(
            name='Subtitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10
        ))
        styles.add(ParagraphStyle(
            name='Normal_Center',
            parent=styles['Normal'],
            alignment=1  # Wyśrodkowany
        ))
        
        # Elementy dokumentu
        elements = []
        
        # Tytuł
        elements.append(Paragraph("Raport diagnostyczny badania USG klaczy", styles['Title']))
        elements.append(Spacer(1, 0.5*cm))
        
        # Data badania
        timestamp = analysis_result.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        elements.append(Paragraph(f"Data badania: {timestamp}", styles['Normal_Center']))
        elements.append(Spacer(1, 0.5*cm))
        
        # Dane obrazu
        image_path = analysis_result.get("image_path", "")
        elements.append(Paragraph(f"Obraz USG: {os.path.basename(image_path)}", styles['Normal']))
        elements.append(Spacer(1, 0.5*cm))
        
        # Obraz oryginalny
        if include_images and os.path.exists(image_path):
            img_width = 400
            img = Image(image_path, width=img_width, height=img_width*0.75)
            elements.append(img)
            elements.append(Spacer(1, 0.3*cm))
            elements.append(Paragraph("Obraz USG", styles['Normal_Center']))
            elements.append(Spacer(1, 0.5*cm))
        
        # Wynik wykrywania ciąży
        pregnancy_result = analysis_result.get("pregnancy", {})
        is_pregnant = pregnancy_result.get("is_pregnant", False)
        confidence = pregnancy_result.get("confidence", 0.0)
        
        elements.append(Paragraph("Wynik analizy obrazu", styles['Subtitle']))
        
        if is_pregnant:
            elements.append(Paragraph(f"<b>Wykryto ciążę</b> (pewność: {confidence:.2f})", styles['Normal']))
            
            # Informacje o dniu ciąży
            day_result = analysis_result.get("day_estimation", {})
            if day_result:
                predicted_day = day_result.get("predicted_day", 0)
                day_confidence = day_result.get("confidence", 0.0)
                
                # Określenie przybliżonej daty krycia i porodu
                current_date = datetime.datetime.strptime(timestamp.split()[0], "%Y-%m-%d")
                breeding_date = current_date - datetime.timedelta(days=predicted_day)
                due_date = breeding_date + datetime.timedelta(days=340)  # Średni czas ciąży klaczy
                
                elements.append(Spacer(1, 0.3*cm))
                elements.append(Paragraph(f"<b>Szacowany dzień ciąży:</b> {predicted_day} (pewność: {day_confidence:.2f})", styles['Normal']))
                elements.append(Paragraph(f"<b>Trymestr:</b> {analysis_result.get('trimester', '-')}", styles['Normal']))
                elements.append(Paragraph(f"<b>Przybliżona data krycia:</b> {breeding_date.strftime('%Y-%m-%d')}", styles['Normal']))
                elements.append(Paragraph(f"<b>Przewidywana data porodu:</b> {due_date.strftime('%Y-%m-%d')}", styles['Normal']))
                
                # Alternatywne estymacje dni ciąży
                top_days = day_result.get("top_5_days", [])
                top_confidences = day_result.get("top_5_confidences", [])
                
                if len(top_days) > 1 and len(top_confidences) > 1:
                    elements.append(Spacer(1, 0.3*cm))
                    elements.append(Paragraph("Alternatywne estymacje dni ciąży:", styles['Normal']))
                    
                    # Tabela z alternatywnymi dniami
                    data = [["Dzień ciąży", "Pewność"]]
                    for day, conf in zip(top_days[1:], top_confidences[1:]):
                        data.append([str(day), f"{conf:.2f}"])
                    
                    t = Table(data, colWidths=[2*cm, 2*cm])
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    
                    elements.append(t)
        else:
            elements.append(Paragraph(f"<b>Nie wykryto ciąży</b> (pewność: {1-confidence:.2f})", styles['Normal']))
        
        # Analiza cech obrazu
        elements.append(Spacer(1, 0.7*cm))
        elements.append(Paragraph("Analiza cech obrazu", styles['Subtitle']))
        
        features = analysis_result.get("image_features", {}).get("basic", {})
        structures = analysis_result.get("image_features", {}).get("structures", {})
        edges = analysis_result.get("image_features", {}).get("edges", {})
        
        # Tabela z cechami obrazu
        data = [
            ["Cecha", "Wartość"],
            ["Średnia intensywność", f"{features.get('mean_intensity', 0):.3f}"],
            ["Kontrast", f"{features.get('contrast', 0):.3f}"],
            ["Entropia", f"{features.get('entropy', 0):.3f}"],
            ["Współczynnik płynu", f"{structures.get('potential_fluid_ratio', 0):.3f}"],
            ["Współczynnik tkanki", f"{structures.get('potential_tissue_ratio', 0):.3f}"],
            ["Siła krawędzi", f"{edges.get('edge_magnitude', 0):.3f}"]
        ]
        
        t = Table(data, colWidths=[4*cm, 3*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(t)
        
        # Wizualizacja analizy
        analysis_vis_path = os.path.join(output_dir, "analysis_visualization.png")
        if include_images and os.path.exists(analysis_vis_path):
            elements.append(Spacer(1, 0.5*cm))
            img = Image(analysis_vis_path, width=doc.width * 0.8, height=doc.width * 0.8 * 0.75)
            elements.append(img)
            elements.append(Spacer(1, 0.3*cm))
            elements.append(Paragraph("Wizualizacja analizy obrazu", styles['Normal_Center']))
        
        # Stopka raportu
        elements.append(Spacer(1, 1*cm))
        elements.append(Paragraph("Raport wygenerowany automatycznie przez system wykrywania ciąży u klaczy.", styles['Normal_Center']))
        elements.append(Paragraph("Wyniki analizy powinny zostać zweryfikowane przez lekarza weterynarii.", styles['Normal_Center']))
        
        # Budowa dokumentu
        doc.build(elements)
        
        log_success(f"Raport PDF został wygenerowany: {report_filename}", log_file)
        
        return report_filename
    
    except Exception as e:
        log_error(f"Błąd podczas generowania raportu PDF: {e}", log_file=log_file)
        return None

def create_batch_report(batch_results, output_dir, log_file=None):
    """Generuje raport PDF podsumowujący analizę wsadową wielu obrazów USG"""
    
    # Nazwa pliku raportu
    report_filename = os.path.join(output_dir, "raport_zbiorczy.pdf")
    
    try:
        # Tworzenie dokumentu PDF
        doc = SimpleDocTemplate(
            report_filename,
            pagesize=A4,
            rightMargin=1.5*cm,
            leftMargin=1.5*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        # Style - POPRAWIONE: getSampleStyleeets() -> getSampleStyleSheet()
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='Title',
            parent=styles['Heading1'],
            fontSize=18,
            alignment=1,
            spaceAfter=12
        ))
        styles.add(ParagraphStyle(
            name='Subtitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10
        ))
        styles.add(ParagraphStyle(
            name='Normal_Center',
            parent=styles['Normal'],
            alignment=1
        ))
        
        # Elementy dokumentu
        elements = []
        
        # Tytuł
        elements.append(Paragraph("Raport zbiorczy analizy obrazów USG klaczy", styles['Title']))
        elements.append(Spacer(1, 0.5*cm))
        
        # Data raportu
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"Data raportu: {timestamp}", styles['Normal_Center']))
        elements.append(Spacer(1, 0.5*cm))
        
        # Statystyki zbiorcze
        pregnant_count = sum(1 for r in batch_results if r.get("pregnancy", {}).get("is_pregnant", False))
        not_pregnant_count = len(batch_results) - pregnant_count
        
        elements.append(Paragraph("Podsumowanie", styles['Subtitle']))
        elements.append(Paragraph(f"Liczba przeanalizowanych obrazów: <b>{len(batch_results)}</b>", styles['Normal']))
        elements.append(Paragraph(f"Liczba wykrytych ciąż: <b>{pregnant_count}</b>", styles['Normal']))
        elements.append(Paragraph(f"Liczba obrazów bez ciąży: <b>{not_pregnant_count}</b>", styles['Normal']))
        
        # Rozkład dni ciąży
        days = [r.get("day_estimation", {}).get("predicted_day", 0) for r in batch_results 
                if r.get("pregnancy", {}).get("is_pregnant", False) and "day_estimation" in r]
        
        if days:
            elements.append(Spacer(1, 0.5*cm))
            elements.append(Paragraph("Statystyki dni ciąży", styles['Subtitle']))
            elements.append(Paragraph(f"Średni dzień ciąży: <b>{np.mean(days):.1f}</b>", styles['Normal']))
            elements.append(Paragraph(f"Mediana dni ciąży: <b>{np.median(days):.1f}</b>", styles['Normal']))
            elements.append(Paragraph(f"Min. dzień ciąży: <b>{min(days)}</b>", styles['Normal']))
            elements.append(Paragraph(f"Maks. dzień ciąży: <b>{max(days)}</b>", styles['Normal']))
            
            # Histogram dni ciąży
            plt.figure(figsize=(8, 6))
            plt.hist(days, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Rozkład dni ciąży')
            plt.xlabel('Dzień ciąży')
            plt.ylabel('Liczba przypadków')
            plt.grid(True, alpha=0.3)
            
            # Zapisz wykres
            histogram_path = os.path.join(output_dir, "days_histogram.png")
            plt.savefig(histogram_path)
            plt.close()
            
            # Dodaj wykres do raportu
            elements.append(Spacer(1, 0.5*cm))
            img = Image(histogram_path, width=doc.width * 0.8, height=doc.width * 0.5)
            elements.append(img)
            elements.append(Spacer(1, 0.3*cm))
            elements.append(Paragraph("Rozkład dni ciąży", styles['Normal_Center']))
        
        # Tabela z wynikami dla każdego obrazu
        elements.append(Spacer(1, 0.7*cm))
        elements.append(Paragraph("Szczegółowe wyniki analizy", styles['Subtitle']))
        
        # Przygotuj dane do tabeli
        data = [["Lp.", "Obraz", "Ciąża", "Pewność", "Dzień ciąży"]]
        
        for i, result in enumerate(batch_results):
            img_name = os.path.basename(result.get("image_path", ""))
            is_pregnant = result.get("pregnancy", {}).get("is_pregnant", False)
            confidence = result.get("pregnancy", {}).get("confidence", 0.0)
            
            if is_pregnant:
                pregnancy_text = "TAK"
                confidence_text = f"{confidence:.2f}"
                day = result.get("day_estimation", {}).get("predicted_day", "-")
                day_text = str(day)
            else:
                pregnancy_text = "NIE"
                confidence_text = f"{1-confidence:.2f}"
                day_text = "-"
            
            data.append([str(i+1), img_name, pregnancy_text, confidence_text, day_text])
        
        # Tworzenie tabeli
        col_widths = [1*cm, 5*cm, 2*cm, 2*cm, 2*cm]
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(t)
        
        # Stopka raportu
        elements.append(Spacer(1, 1*cm))
        elements.append(Paragraph("Raport wygenerowany automatycznie przez system wykrywania ciąży u klaczy.", styles['Normal_Center']))
        elements.append(Paragraph("Wyniki analizy powinny zostać zweryfikowane przez lekarza weterynarii.", styles['Normal_Center']))
        
        # Budowa dokumentu
        doc.build(elements)
        
        log_success(f"Zbiorczy raport PDF został wygenerowany: {report_filename}", log_file)
        
        return report_filename
    
    except Exception as e:
        log_error(f"Błąd podczas generowania zbiorczego raportu PDF: {e}", log_file=log_file)
        return None
