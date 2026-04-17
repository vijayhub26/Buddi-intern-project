"""
PDF renderer using PyMuPDF (fitz).
Renders each page of a PDF to a high-DPI numpy array for OCR processing.
"""
import fitz  # PyMuPDF
import numpy as np
from typing import Generator, Tuple


def render_pdf_pages(
    pdf_path: str,
    dpi: int = 300,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Render each page of a PDF to a numpy array.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        dpi: Resolution for rendering (300 DPI recommended for OCR accuracy).

    Yields:
        (page_number, image_array) tuples.
        page_number is 1-indexed.
        image_array is a BGR numpy array (H x W x 3).
    """
    zoom = dpi / 300  # PyMuPDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    doc = fitz.open(pdf_path)
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            # pix.samples is raw RGB bytes
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            # Convert RGB → BGR for OpenCV compatibility
            img_bgr = img_array[:, :, ::-1].copy()
            yield page_index + 1, img_bgr
    finally:
        doc.close()


def page_count(pdf_path: str) -> int:
    """Return the total number of pages in the PDF."""
    with fitz.open(pdf_path) as doc:
        return len(doc)
