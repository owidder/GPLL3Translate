from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
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
    normal_style = styles['Normal']
    normal_style.wordWrap = 'CJK'  # Ermöglicht Umbruch für lange Wörter
    
    # Wandle Zelleninhalte in Paragraphen um
    formatted_data = []
    for row in data:
        formatted_row = [Paragraph(str(cell), normal_style) for cell in row]
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
    # Beispieldaten für die Tabelle mit längeren Texten
    table_data = [
        ['Spalte 1', 'Spalte 2', 'Spalte 3'],
        ['Dies ist ein sehr langer Text, der umgebrochen werden sollte.', 'Kurzer Text', 'Noch ein langer Text, der ebenfalls umgebrochen werden sollte.'],
        ['Zeile 2, Zelle 1', 'Ein mittelanger Text, der vielleicht umgebrochen wird.', 'Zeile 2, Zelle 3'],
        ['Zeile 3, Zelle 1', 'Zeile 3, Zelle 2', 'Zeile 3, Zelle 3'],
    ]
    
    create_pdf_with_table('beispiel_tabelle_mit_umbruch.pdf', table_data)
