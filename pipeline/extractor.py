"""
End-to-end extraction orchestrator.
Ties together: PDF rendering → preprocessing → OCR → layout reconstruction → (optional) LLM post-correction.
"""
from typing import List, Tuple, Optional
from pipeline.pdf_renderer import render_pdf_pages, page_count
from pipeline.ocr_engine import get_ocr_engine
from pipeline.layout_reconstructor import reconstruct_layout
from preprocessing.cleaner import clean_image
import numpy as np
import re
import wordninja
from pipeline.utils import spell
_spell = spell

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
def fix_clumping(text: str) -> str:
    """Split run-together words using wordninja while strictly preserving layout spaces.
    Protected tokens (tech specs, brand names) are stashed before splitting.
    """
    # Pre-protect storage/speed units and brand names from being split
    _PROTECT_RE = [
        # re.compile(r'\b(\d+(?:\.\d+)?)(G[Hh]z|M[Hh]z|[KMGT]B|[Hh]z)\b'),
        # re.compile(r'\b[iI][3579](?:-\d+)?\b'),
        # re.compile(r'\bTHREADRIPPER\b', re.IGNORECASE),
    ]
    _stash: dict = {}
    _counter = [0]

    def _stash_fn(m: re.Match) -> str:
        key = f'\xA7{_counter[0]}\xA7'  # § delimiters — safe from alpha/digit split rules
        _counter[0] += 1
        _stash[key] = m.group(0)
        return key

    for pat in _PROTECT_RE:
        text = pat.sub(_stash_fn, text)

    # Split by any whitespace, but capture the whitespace to preserve layout gaps
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

    text = "".join(parts)

    # Restore protected tokens
    for key, original in _stash.items():
        text = text.replace(key, original)

    return text

def extract_text_from_pdf(
    pdf_path: str,
    dpi: int = 300,
    min_confidence: float = 0.0,
    pages: Optional[List[int]] = None,
    progress_callback=None,
    exclude_patterns: Optional[List[str]] = None,
    ignore_symbols: bool = False,
    post_correct: bool = False,
) -> Tuple[List[PageResult], str]:
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
        exclude_patterns:  Optional list of regex patterns to filter out of the result.

    Returns:
        (page_results, full_text)
    """
    engine = get_ocr_engine()

    total = page_count(pdf_path)
    page_results: List[PageResult] = []

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

    # Combine all pages with a clear separator
    separator = "\n\n" + ("─" * 60) + "\n"
    full_text = separator.join(
        f"[Page {pn}]\n{text}" for pn, text in page_results
    )
    
    if ignore_symbols:
        full_text = strip_symbols(full_text)

    return page_results, full_text