from typing import List, Dict
import pdfplumber


def extract_pages(pdf_file) -> List[Dict]:
    """Extract text per page from a PDF file."""
    pages = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page_num": i, "text": text.strip()})
    return pages
