from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


def create_pdf_with_html_text(filename):
    # Create a PDF document
    pdf = SimpleDocTemplate(filename, pagesize=letter)

    # Define a style for the Paragraph
    styles = getSampleStyleSheet()
    styleN = styles['Normal']

    # Define data for the table with HTML formatted text
    data = [
        [Paragraph('<b>Header 1</b>', styleN), Paragraph('<b>Header 2</b>', styleN),
         Paragraph('<b>Header 3</b>', styleN)],
        [Paragraph(
            '<i>This is a <u>longer description</u> in column 1 with <font color="red">HTML</font> formatting</i>',
            styleN),
         Paragraph('Row 1, Col 2', styleN), Paragraph('Row 1, Col 3', styleN)],
        [Paragraph('Row 2, Col 1', styleN),
         Paragraph('<b>Another long text</b> that needs <u>wrapping</u> in column 2', styleN),
         Paragraph('Row 2, Col 3', styleN)],
        [Paragraph('Row 3, Col 1', styleN), Paragraph('Row 3, Col 2', styleN),
         Paragraph(
             '<font size="12">This is a very long text in column 3 which <font color="blue"><b>should wrap appropriately</b></font></font>',
             styleN)],
    ]

    # Create the table
    table = Table(data)

    # Set the style for the table
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),  # Ensures the text is aligned to the top of the cell
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('FONTSIZE', (0, 0), (-1, -1), 10),  # Adjust the font size if needed
    ])
    table.setStyle(style)

    # Adjust the column widths to fit the page width
    table_width = pdf.pagesize[0] - pdf.leftMargin - pdf.rightMargin
    num_columns = len(data[0])
    col_width = table_width / num_columns
    table._argW = [col_width] * num_columns

    # Build the PDF
    elements = [table]
    pdf.build(elements)


# Call the function to create the PDF
create_pdf_with_html_text("table_html_text_example.pdf")
