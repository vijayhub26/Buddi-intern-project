"""
End-to-end extraction orchestrator.
Ties together: PDF rendering → preprocessing → OCR → layout reconstruction.
"""
from typing import List, Tuple, Optional
from pipeline.pdf_renderer import render_pdf_pages, page_count
from pipeline.ocr_engine import RapidOCREngine
from pipeline.layout_reconstructor import reconstruct_layout
from preprocessing.cleaner import clean_image


PageResult = Tuple[int, str]  # (page_number, reconstructed_text)


def extract_text_from_pdf(
    pdf_path: str,
    dpi: int = 300,
    deskew: bool = True,
    min_confidence: float = 0.0,
    pages: Optional[List[int]] = None,
    progress_callback=None,
) -> Tuple[List[PageResult], str]:
    """
    Extract text from an image-based PDF preserving layout.

    Args:
        pdf_path:          Path to the input PDF file.
        dpi:               Rendering resolution (300 DPI recommended).
        deskew:            Whether to apply deskew correction.
        min_confidence:    Discard OCR results below this confidence score.
        pages:             Optional list of 1-indexed page numbers to process.
                           If None, all pages are processed.
        progress_callback: Optional callable(page_num, total_pages) for progress.

    Returns:
        (page_results, full_text)
        page_results: list of (page_number, page_text) tuples
        full_text:    all pages joined with page-break separators
    """
    engine = RapidOCREngine()
    total = page_count(pdf_path)
    page_results: List[PageResult] = []

    for page_num, img in render_pdf_pages(pdf_path, dpi=dpi):
        if pages is not None and page_num not in pages:
            continue

        if progress_callback:
            progress_callback(page_num, total)

        # 1. Preprocess with OpenCV
        cleaned = clean_image(img, deskew_enabled=deskew)

        # 2. Run RapidOCR
        ocr_results = engine.recognize(cleaned)

        # 3. Filter by confidence
        if min_confidence > 0.0:
            ocr_results = [r for r in ocr_results if r[2] >= min_confidence]

        # 4. Reconstruct layout-preserving text
        page_text = reconstruct_layout(ocr_results)
        page_results.append((page_num, page_text))

    # Combine all pages with a clear separator
    separator = "\n\n" + ("─" * 60) + "\n"
    full_text = separator.join(
        f"[Page {pn}]\n{text}" for pn, text in page_results
    )

    return page_results, full_text
