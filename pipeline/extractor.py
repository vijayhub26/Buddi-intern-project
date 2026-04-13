"""
End-to-end extraction orchestrator.
Ties together: PDF rendering → preprocessing → OCR → layout reconstruction → (optional) LLM post-correction.
"""
from typing import List, Tuple, Optional, Dict, Any
from pipeline.pdf_renderer import render_pdf_pages, page_count
from pipeline.ocr_engine import get_ocr_engine
from pipeline.layout_reconstructor import reconstruct_layout
from preprocessing.cleaner import clean_image
import numpy as np
import re
import wordninja
from spellchecker import SpellChecker
_spell = SpellChecker()

# Lazy-loaded so startup time isn't affected when post_correct=False
_corrector = None
def _get_corrector():
    global _corrector
    if _corrector is None:
        from pipeline.post_corrector import PostCorrector
        _corrector = PostCorrector()
    return _corrector

def strip_symbols(text: str) -> str:
    """
    Keep only letters, digits, and whitespace. 
    Preserves decimal points only if they are between digits (e.g., '10.50').
    """
    if not text: return ""
    # 1. First, remove everything that isn't Alphanumeric, Space, or Dot
    text = re.sub(r'[^a-zA-Z0-9\s.]', '', text)
    # 2. Then, remove dots that are NOT surrounded by digits on both sides
    #    This protects prices/quantities but removes sentence-ending periods.
    text = re.sub(r'(?<!\d)\.|\.(?!\d)', '', text)
    return text

PageResult = Tuple[int, str]  # (page_number, reconstructed_text)

# Raw per-page data used for the searchable PDF overlay
PageData = Dict[str, Any]  # {page_num, image, ocr_results, img_w, img_h}
def fix_clumping(text: str) -> str:
    """Split run-together words using wordninja while strictly preserving layout spaces."""
    import re
    # Split by any whitespace, but capture the whitespace to keep it
    parts = re.split(r'(\s+)', text)
    for i in range(0, len(parts), 2):
        tok = parts[i]
        if not tok: 
            continue
            
        clean = tok.strip('.,;:!?()\'"')
        # Only attempt to split long pure-alpha tokens not in dictionary
        if len(clean) >= 7 and clean.isalpha() and clean.lower() not in _spell:
            split_parts = wordninja.split(clean)
            if len(split_parts) > 1:
                parts[i] = tok.replace(clean, " ".join(split_parts))
                
    return "".join(parts)

def extract_text_from_pdf(
    pdf_path: str,
    dpi: int = 300,
    min_confidence: float = 0.0,
    pages: Optional[List[int]] = None,
    progress_callback=None,
    return_raw: bool = False,
    exclude_patterns: Optional[List[str]] = None,
    ignore_symbols: bool = False,
    post_correct: bool = False,
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
        exclude_patterns:  Optional list of regex patterns to filter out of the result.

    Returns:
        If return_raw=False: (page_results, full_text)
        If return_raw=True:  (page_results, full_text, pages_data)
    """
    engine = get_ocr_engine()

    total = page_count(pdf_path)
    page_results: List[PageResult] = []
    pages_data: List[PageData] = []

    for page_num, img in render_pdf_pages(pdf_path, dpi=dpi):
        if pages is not None and page_num not in pages:
            continue

        if progress_callback:
            progress_callback(page_num, total)

        img_h, img_w = img.shape[:2]

       # 1. Preprocess
        cleaned = clean_image(img)
        img_h, img_w = cleaned.shape[:2]  # ← move here, was before clean_image()

        # 2. Run PaddleOCR
        ocr_results = engine.recognize(cleaned)

        # 3. Filter by confidence
        if min_confidence > 0.0:
            ocr_results = [r for r in ocr_results if r[2] >= min_confidence]

        # 4. Reconstruct layout-preserving text
        page_text = reconstruct_layout(
            ocr_results, 
            image=cleaned,
            page_width=img_w, 
            exclude_patterns=exclude_patterns,
        )
        page_results.append((page_num, page_text))
        page_text = fix_clumping(page_text)

        # 5. LLM post-correction (optional)
        if post_correct:
            page_text = _get_corrector().correct(page_text, verbose=True)
            # Update page_results with corrected text
            page_results[-1] = (page_num, page_text)

        if ignore_symbols:
            page_text = strip_symbols(page_text)
        # 6. Store raw data for overlay (original image + OCR boxes)
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
    
    if ignore_symbols:
        full_text = strip_symbols(full_text)

    if return_raw:
        return page_results, full_text, pages_data

    return page_results, full_text