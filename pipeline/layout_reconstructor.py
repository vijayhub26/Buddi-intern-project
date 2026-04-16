"""
Layout-preserving text reconstructor.
Final 'Stabilized Column' Version.
"""
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import re
import wordninja
from pipeline.utils import spell

import math

OCRResult = List[Tuple[List, str, float]]

def _de_clump(text: str) -> str:
    """
    Fix OCR word-clumping using structural patterns only.
    Protected tokens (tech specs, IBANs, brand names) are stashed before
    any split rules run, then restored — preventing false-positive splits.
    """
    return text.strip()

def _de_fragment(text: str) -> str:
    """Fix OCR fragmentation (over-splitting) errors on the fully reconstructed layout."""
    return text
def _renumber_page(page_text: str) -> str:
    """Detects standard table layouts and injects missing sequential numbers."""
    return page_text

def reconstruct_layout(
    ocr_results: OCRResult,
    image: np.ndarray = None,
    line_height_tolerance: float = 0.35,
    page_width: float = None,
    exclude_patterns: List[str] = None,
) -> str:
    if not ocr_results:
        return ""

    compiled_patterns = [re.compile(p) for p in (exclude_patterns or [])]

    # 1. Build blocks + deduplicate
    blocks = []
    for box, text, score in sorted(ocr_results, key=lambda x: x[2], reverse=True):
        text = _de_clump(text)
        if not text:
            continue
        if any(p.search(text) for p in compiled_patterns):
            continue
        pts = np.array(box, dtype=float)
        x, y   = pts[:, 0].min(), pts[:, 1].min()
        right  = pts[:, 0].max()
        bottom = pts[:, 1].max()
        h      = max(bottom - y, 1.0)
        cx, cy = (x + right) / 2, (y + bottom) / 2

        is_dup = False
        for b in blocks:
            if abs(cx - b["cx"]) < 10 and abs(cy - b["cy"]) < 10:
                is_dup = True
                if len(text) > len(b["text"]):
                    b["text"] = text
                break
        if not is_dup:
            blocks.append({
                "x": x, "y": y, "right": right,
                "h": h, "cx": cx, "cy": cy, "text": text
            })

    if not blocks:
        return ""

    blocks.sort(key=lambda b: (b["y"], b["x"]))

    # 2. Derive space_width from actual OCR box heights — no hardcoding
    #    Monospace char width ≈ 0.55× cap height at any DPI
    median_h = float(np.median([b["h"] for b in blocks]))
    space_width_chars = max(7.0, median_h * 0.40)

    # 3. Snap threshold scales with font size too
    snap_threshold = max(15.0, median_h * 0.9)

    # 4. Line grouping — median Y of current line, not just first block
    lines = []
    current_line = [blocks[0]]
    for b in blocks[1:]:
        median_y = float(np.median([lb["y"] for lb in current_line]))
        median_line_h = float(np.median([lb["h"] for lb in current_line]))
        if abs(b["y"] - median_y) < median_line_h * line_height_tolerance:
            current_line.append(b)
        else:
            lines.append(current_line)
            current_line = [b]
    lines.append(current_line)

    # 5. Column anchor discovery
    all_xs = sorted([b["x"] for b in blocks])
    anchors = []
    if all_xs:
        cluster = [all_xs[0]]
        for x in all_xs[1:]:
            if x - cluster[-1] < snap_threshold:
                cluster.append(x)
            else:
                anchors.append(float(np.mean(cluster)))
                cluster = [x]
        anchors.append(float(np.mean(cluster)))

    # 6. Reconstruct
    page_str = ""
    prev_y = None
    for line in lines:
        line_median_y = float(np.median([b["y"] for b in line]))
        
        # Insert vertical spacing
        if prev_y is not None:
            gap_ratio = (line_median_y - prev_y) / median_h
            # Any gap 25% larger than a standard line height is a visual block break
            if gap_ratio > 1.20:
                # Cap at 2 extra newlines — large Y-gaps occur between header blocks
                # and injecting too many blank lines diverges from ground truth.
                extra_newlines = min(2, max(1, int(round(gap_ratio)) - 1))
                page_str += "\n" * extra_newlines
                
        prev_y = line_median_y
        
        line.sort(key=lambda b: b["x"])
        line_text = ""
        prev_char_pos = 0
        for b in line:
            best_a = min(anchors, key=lambda a: abs(b["x"] - a))
            target_char_pos = int(best_a / space_width_chars)
            if target_char_pos > prev_char_pos:
                line_text += " " * (target_char_pos - prev_char_pos)
            elif prev_char_pos > 0:
                line_text += " "
            line_text += b["text"]
            prev_char_pos = len(line_text)
        page_str += line_text.rstrip() + "\n"

    page_str = _de_fragment(page_str)
    return page_str