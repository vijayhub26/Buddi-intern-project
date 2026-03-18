"""
pdf_text_extractor.py
~~~~~~~~~~~~~~~~~~~~~
Extract the embedded text layer from a searchable PDF and reconstruct the
original spatial layout as a plain-text string.

Works on PDFs produced by create_searchable_pdf() (pipeline/pdf_writer.py),
which stores each OCR word/block as an invisible text span at its exact
bounding-box position.  PyMuPDF's word-level extraction gives us those
coordinates back so we can rebuild the visual layout.

Public API
----------
extract_text_from_searchable_pdf(pdf_path, pages=None) -> str
"""

from __future__ import annotations

import fitz  # PyMuPDF (already required: pymupdf>=1.23.0)
import numpy as np
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Type alias for a single extracted word record
# (x0, y0, x1, y1, word_text, block_no, line_no, word_no)
# ---------------------------------------------------------------------------
WordRecord = Tuple[float, float, float, float, str, int, int, int]


# ---------------------------------------------------------------------------
# Layout reconstruction (operates on PDF-point coordinates)
# ---------------------------------------------------------------------------

def _reconstruct_layout(
    words: List[WordRecord],
    line_height_tolerance: float = 0.5,
    space_width_pts: float = 6.0,
) -> str:
    """
    Reconstruct a layout-preserving text string from a list of word records.

    Args:
        words:                 List of (x0,y0,x1,y1,text,…) from fitz word extraction.
        line_height_tolerance: Fraction of average word height for Y-grouping threshold.
        space_width_pts:       Approximate width (in PDF points) of one space character.
                               Used to decide how many spaces to insert between words.

    Returns:
        Multi-line string that mirrors the spatial layout of the page.
    """
    if not words:
        return ""

    # Build structured blocks from word records
    blocks: List[dict] = []
    for x0, y0, x1, y1, text, *_ in words:
        text = text.strip()
        if not text:
            continue
        blocks.append({
            "x":    x0,
            "y":    y0,
            "right": x1,
            "h":    max(y1 - y0, 1.0),
            "text": text,
        })

    if not blocks:
        return ""

    # Sort top-to-bottom, left-to-right
    blocks.sort(key=lambda b: (b["y"], b["x"]))

    # Average height — used as grouping threshold
    avg_h = float(np.mean([b["h"] for b in blocks]))
    _threshold = avg_h * line_height_tolerance  # kept for reference; overlap used below

    # --- Group blocks into visual lines using vertical overlap ---
    lines: List[List[dict]] = []
    for block in blocks:
        placed = False
        b_top = block["y"]
        b_bot = b_top + block["h"]

        for line in lines:
            l_top = min(b["y"] for b in line)
            l_bot = max(b["y"] + b["h"] for b in line)

            overlap = max(0.0, min(l_bot, b_bot) - max(l_top, b_top))
            min_h   = min(l_bot - l_top, block["h"])

            if min_h > 0 and (overlap / min_h) > 0.4:
                line.append(block)
                placed = True
                break

        if not placed:
            lines.append([block])

    # Sort each line left-to-right, then all lines top-to-bottom
    for i, line in enumerate(lines):
        lines[i] = sorted(line, key=lambda b: b["x"])
    lines.sort(key=lambda line: float(np.mean([b["y"] for b in line])))

    # --- Assemble each line with gap-proportional spaces ---
    text_lines: List[str] = []
    for line_blocks in lines:
        row = line_blocks[0]["text"]
        for i in range(1, len(line_blocks)):
            prev = line_blocks[i - 1]
            curr = line_blocks[i]
            gap_pts  = max(0.0, curr["x"] - prev["right"])
            n_spaces = max(1, int(round(gap_pts / space_width_pts)))
            row += " " * n_spaces + curr["text"]
        text_lines.append(row)

    return "\n".join(text_lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_text_from_searchable_pdf(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    space_width_pts: float = 6.0,
) -> str:
    """
    Open a searchable PDF and reconstruct its text layout as a plain string.

    Args:
        pdf_path:        Path to the searchable PDF file.
        pages:           Optional list of 1-indexed page numbers to include.
                         If None, all pages are processed.
        space_width_pts: Controls how many spaces represent a horizontal gap.
                         Decrease for denser text, increase for wider columns.

    Returns:
        A single string with all pages separated by a line of dashes.
        Each page section is headed with "[Page N]".
    """
    doc = fitz.open(pdf_path)
    page_texts: List[str] = []

    try:
        total = len(doc)
        for page_index in range(total):
            page_num = page_index + 1

            if pages is not None and page_num not in pages:
                continue

            page = doc[page_index]

            # get_text("words") returns:
            #   [(x0, y0, x1, y1, "word", block_no, line_no, word_no), ...]
            # Coordinates are in PDF points (origin = top-left for fitz).
            word_records: List[WordRecord] = page.get_text("words")  # type: ignore[assignment]

            page_text = _reconstruct_layout(
                word_records,
                space_width_pts=space_width_pts,
            )
            page_texts.append((page_num, page_text))

    finally:
        doc.close()

    # Join pages with a clear separator
    separator = "\n\n" + ("─" * 60) + "\n"
    full_text = separator.join(
        f"[Page {pn}]\n{text}" for pn, text in page_texts
    )
    return full_text
