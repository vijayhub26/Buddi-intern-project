"""
End-to-end extraction orchestrator.
Ties together: PDF rendering → preprocessing → OCR → layout reconstruction.
"""
from typing import List, Tuple, Optional, Dict, Any
from pipeline.pdf_renderer import render_pdf_pages, page_count
from pipeline.ocr_engine import RapidOCREngine
from pipeline.layout_reconstructor import reconstruct_layout
from preprocessing.cleaner import clean_image
import numpy as np


PageResult = Tuple[int, str]  # (page_number, reconstructed_text)

# Raw per-page data used for the searchable PDF overlay
PageData = Dict[str, Any]  # {page_num, image, ocr_results, img_w, img_h}


def extract_text_from_pdf(
    pdf_path: str,
    dpi: int = 200,
    deskew: bool = True,
    min_confidence: float = 0.0,
    pages: Optional[List[int]] = None,
    progress_callback=None,
    return_raw: bool = False,
) -> Tuple:
    """
    Extract text from an image-based PDF preserving layout.

    Args:
        pdf_path:          Path to the input PDF file.
        dpi:               Rendering resolution (200 DPI recommended for digital).
        deskew:            Whether to apply deskew correction.
        min_confidence:    Discard OCR results below this confidence score.
        pages:             Optional list of 1-indexed page numbers to process.
                           If None, all pages are processed.
        progress_callback: Optional callable(page_num, total_pages) for progress.
        return_raw:        If True, also return pages_data for searchable PDF overlay.

    Returns:
        If return_raw=False: (page_results, full_text)
        If return_raw=True:  (page_results, full_text, pages_data)
    """
    engine = RapidOCREngine()
    total = page_count(pdf_path)
    page_results: List[PageResult] = []
    pages_data: List[PageData] = []

    for page_num, img in render_pdf_pages(pdf_path, dpi=dpi):
        if pages is not None and page_num not in pages:
            continue

        if progress_callback:
            progress_callback(page_num, total)

        img_h, img_w = img.shape[:2]

        # 1. Preprocess with OpenCV
        cleaned = clean_image(img, deskew_enabled=deskew)

        # 2. Run RapidOCR
        ocr_results = engine.recognize(cleaned)

        # 3. Filter by confidence
        if min_confidence > 0.0:
            ocr_results = [r for r in ocr_results if r[2] >= min_confidence]

        # 4. Reconstruct layout-preserving text
        page_text = reconstruct_layout(ocr_results, page_width=img_w)
        page_results.append((page_num, page_text))

        # 5. Store raw data for overlay (original image + OCR boxes)
        if return_raw:
            pages_data.append({
                "page_num": page_num,
                "image": img,          # original BGR image (for background)
                "ocr_results": ocr_results,
                "img_w": img_w,
                "img_h": img_h,
                "dpi": dpi,
            })

    # Combine all pages with a clear separator
    separator = "\n\n" + ("─" * 60) + "\n"
    full_text = separator.join(
        f"[Page {pn}]\n{text}" for pn, text in page_results
    )

    if return_raw:
        return page_results, full_text, pages_data

    return page_results, full_text
