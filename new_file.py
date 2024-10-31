import pdfkit
import os

def html_to_pdf(html_file, pdf_file):
    # Get the absolute path of the HTML file
    html_path = os.path.abspath(html_file)
    
    # Configure pdfkit options
    options = {
        'page-size': 'A4',
        'margin-top': '0mm',
        'margin-right': '0mm',
        'margin-bottom': '0mm',
        'margin-left': '0mm',
        'encoding': "UTF-8",
        'no-outline': None
    }

    # Convert HTML to PDF
    pdfkit.from_file(html_path, pdf_file, options=options)

if __name__ == "__main__":
    html_file = "aider/01_co2_pricing.json_table.de_sv.html"
    pdf_file = "output.pdf"
    
    html_to_pdf(html_file, pdf_file)
    print(f"PDF file created: {pdf_file}")
