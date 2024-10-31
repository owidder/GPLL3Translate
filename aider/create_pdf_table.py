import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm

def create_pdf_with_table(filename, data):
    """
    Erzeugt ein A4-PDF-Dokument mit einer Tabelle, die die gesamte Breite einnimmt.
    Lange Texte werden automatisch umgebrochen.
    
    :param filename: Name der zu erstellenden PDF-Datei
    :param data: Liste von Listen, die die Tabellenzeilen repräsentieren
    """
    # Erstelle ein neues PDF-Dokument
    doc = SimpleDocTemplate(filename, pagesize=A4)
    
    # Berechne die verfügbare Breite (A4-Breite minus Ränder)
    available_width = A4[0] - 2*cm
    
    # Definiere Stile
    styles = getSampleStyleSheet()
    html_style = ParagraphStyle('HTMLStyle', parent=styles['Normal'])
    html_style.wordWrap = 'CJK'  # Ermöglicht Umbruch für lange Wörter
    html_style.allowWidows = 0
    html_style.allowOrphans = 0
    
    # Wandle Zelleninhalte in Paragraphen um und erlaube HTML-Formatierung
    formatted_data = []
    for row in data:
        formatted_row = [Paragraph(cell, html_style) for cell in row]
        formatted_data.append(formatted_row)
    
    # Erstelle die Tabelle mit automatischen Zeilenhöhen
    col_widths = [available_width/len(data[0])]*len(data[0])
    table = Table(formatted_data, colWidths=col_widths)
    
    # Definiere den Tabellenstil
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ])
    
    table.setStyle(style)
    
    # Baue das Dokument
    elements = [table]
    doc.build(elements)

# Beispielaufruf
if __name__ == "__main__":
    # Beispieldaten für die Tabelle mit HTML-formatierten Texten und <span> Elementen
    table_data = [
        ['<span style="font-weight: bold;">Spalte 1</span>', '<span style="font-weight: bold;">Spalte 2</span>', '<span style="font-weight: bold;">Spalte 3</span>'],
        ['Dies ist ein <span style="font-style: italic;">sehr langer Text</span>, der umgebrochen werden sollte.', '<span style="color: red;">Kurzer Text</span>', 'Noch ein <span style="text-decoration: underline;">langer Text</span>, der ebenfalls umgebrochen werden sollte.'],
        ['Zeile 2, <span style="font-weight: bold;">Zelle 1</span>', 'Ein <span style="font-style: italic;">mittelanger</span> Text, der vielleicht umgebrochen wird.', 'Zeile 2, <span style="color: blue;">Zelle 3</span>'],
        ['<span style="font-weight: bold; font-style: italic; text-decoration: underline;">Formatierter Text</span>', '<span style="font-size: 14px;">Größerer Text</span>', '<span style="font-family: Courier;">Monospace-Schrift</span>'],
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, 'beispiel_tabelle_mit_umbruch.pdf')
    create_pdf_with_table(pdf_path, table_data)

# Zusätzliche Beispiele für komplexere Span-Verwendung
def complex_span_example():
    complex_data = [
        ['<span style="font-weight: bold; color: #FF5733;">Komplexe Formatierung</span>'],
        ['<span style="background-color: #FFFFCC; padding: 5px; border: 1px solid #000000;">Hintergrund und Rahmen</span>'],
        ['<span style="font-size: 16px; font-family: Arial, sans-serif;">Benutzerdefinierte Schriftart und Größe</span>'],
        ['Normaler Text mit <span style="font-weight: bold; color: blue;">fettgedrucktem blauen</span> und <span style="font-style: italic; color: green;">kursivem grünen</span> Text'],
    ]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, 'komplexe_span_beispiele.pdf')
    create_pdf_with_table(pdf_path, complex_data)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, 'beispiel_tabelle_mit_umbruch.pdf')
    create_pdf_with_table(pdf_path, table_data)
    complex_span_example()
