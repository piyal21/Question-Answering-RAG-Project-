import fitz
from src.config import PDF_PATH


def extract_raw_pdf_text(pdf_path=PDF_PATH):
    """Extract raw text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        texts.append(page.get_text())
    doc.close()
    return "\n".join(texts)
