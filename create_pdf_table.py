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
    # Beispieldaten für die Tabelle mit HTML-formatierten Texten
    table_data = [
        ['<b>Spalte 1</b>', '<b>Spalte 2</b>', '<b>Spalte 3</b>'],
        ['Dies ist ein <i>sehr langer Text</i>, der umgebrochen werden sollte.', '<font color="red">Kurzer Text</font>', 'Noch ein <u>langer Text</u>, der ebenfalls umgebrochen werden sollte.'],
        ['Zeile 2, <b>Zelle 1</b>', 'Ein <i>mittelanger</i> Text, der vielleicht umgebrochen wird.', 'Zeile 2, <font color="blue">Zelle 3</font>'],
        ['<b><i><u>Formatierter Text</u></i></b>', '<font size="14">Größerer Text</font>', '<font face="Courier">Monospace-Schrift</font>'],
    ]
    
    create_pdf_with_table('beispiel_tabelle_mit_umbruch.pdf', table_data)
