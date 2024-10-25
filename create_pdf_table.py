from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.units import cm

def create_pdf_with_table(filename, data):
    """
    Erzeugt ein A4-PDF-Dokument mit einer Tabelle, die die gesamte Breite einnimmt.
    
    :param filename: Name der zu erstellenden PDF-Datei
    :param data: Liste von Listen, die die Tabellenzeilen repräsentieren
    """
    # Erstelle ein neues PDF-Dokument
    doc = SimpleDocTemplate(filename, pagesize=A4)
    
    # Berechne die verfügbare Breite (A4-Breite minus Ränder)
    available_width = A4[0] - 2*cm
    
    # Erstelle die Tabelle
    table = Table(data, colWidths=[available_width/len(data[0])]*len(data[0]))
    
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
    # Beispieldaten für die Tabelle
    table_data = [
        ['Spalte 1', 'Spalte 2', 'Spalte 3'],
        ['Zeile 1, Zelle 1', 'Zeile 1, Zelle 2', 'Zeile 1, Zelle 3'],
        ['Zeile 2, Zelle 1', 'Zeile 2, Zelle 2', 'Zeile 2, Zelle 3'],
        ['Zeile 3, Zelle 1', 'Zeile 3, Zelle 2', 'Zeile 3, Zelle 3'],
    ]
    
    create_pdf_with_table('beispiel_tabelle.pdf', table_data)
