"""
Layout-preserving text reconstructor.
Final 'Stabilized Column' Version.
"""
from typing import List, Tuple, Optional
import numpy as np
import re
import wordninja

OCRResult = List[Tuple[List, str, float]]

def _de_clump(text: str) -> str:
    """
    Fix OCR word-clumping using structural patterns only.
    No hardcoded vocabulary — works on any document.
    """
    text = text.strip()
    if not text:
        return ""

    # 1. CamelCase split: 'invoiceDate' -> 'invoice Date'
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # 2. Colon spacing: 'SAC:998439' -> 'SAC: 998439'
    text = re.sub(r'(:)([^\s])', r'\1 \2', text)

    # 2.5 Punctuation spacing: 'Cancel.See' -> 'Cancel. See' (ignores digits like 18,000)
    text = re.sub(r'([.?!,;])([A-Za-z])', r'\1 \2', text)

    # 3. Alpha→Digit boundary: 'Invoice51109' -> 'Invoice 51109'
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)

    # 4. Digit→Alpha boundary: '2013Invoice' -> '2013 Invoice'
    text = re.sub(r'(\d)([A-Za-z]{2,})', r'\1 \2', text)

    # 5. Long-Alpha split: 'detailedinstructions' -> 'detailed instructions'
    #    Run wordninja on tokens that are purely alphabetical and long
    tokens = text.split()
    new_tokens = []
    for tok in tokens:
        clean_tok = tok.strip('.,;:!?()\'"')
        if len(clean_tok) > 15 and clean_tok.isalpha():
            # wordninja splits the word optimally based on English frequencies
            # if it was TitleCaseRunOn, CamelCase handled it mostly, but this catches lowercase run-ons.
            new_tokens.append(" ".join(wordninja.split(tok)))
        else:
            new_tokens.append(tok)
    text = " ".join(new_tokens)

    # 6. OCR misspelling correction for 'I' -> 'l'
    text = re.sub(r'\blf\b', 'If', text)

    return text

def reconstruct_layout(
    ocr_results: OCRResult,
    image: np.ndarray = None,
    line_height_tolerance: float = 0.6,
    space_width_chars: float = 18.0, # Calibrated for 300 DPI
    page_width: float = None,
    total_columns: int = 120,
    exclude_patterns: List[str] = None,
) -> str:
    if not ocr_results: return ""

    # 1. Clean and deduplicate (60px)
    blocks = []
    sorted_res = sorted(ocr_results, key=lambda x: x[2], reverse=True)
    for box, text, score in sorted_res:
        text = _de_clump(text)
        if not text: continue
        pts = np.array(box, dtype=float)
        x, y = pts[:, 0].min(), pts[:, 1].min()
        right, bottom = pts[:, 0].max(), pts[:, 1].max()
        h = bottom - y
        cx, cy = (x + right)/2, (y + bottom)/2
        
        is_dup = False
        for b in blocks:
            if np.sqrt((cx - b["cx"])**2 + (cy - b["cy"])**2) < 60:
                is_dup = True
                if len(text) > len(b["text"]): b["text"] = text
                break
        if not is_dup:
            blocks.append({"x": x, "y": y, "right": right, "h": h, "cx": cx, "cy": cy, "text": text})

    if not blocks: return ""
    blocks.sort(key=lambda b: (b["y"], b["x"]))

    # 2. Extract Lines
    lines = []
    current_line = [blocks[0]]
    for b in blocks[1:]:
        if abs(b["y"] - current_line[0]["y"]) < current_line[0]["h"] * line_height_tolerance:
            current_line.append(b)
        else:
            lines.append(current_line)
            current_line = [b]
    lines.append(current_line)

    # 3. Discover Column Anchors (Snap X positions)
    # Find natural gutters across the whole page to force vertical alignment
    all_xs = sorted([b["x"] for b in blocks])
    anchors = []
    if all_xs:
        curr_a = all_xs[0]
        cluster = [all_xs[0]]
        for x in all_xs[1:]:
            if x - curr_a < 35: # Snap threshold for columns
                cluster.append(x)
            else:
                anchors.append(np.mean(cluster))
                curr_a = x
                cluster = [x]
        anchors.append(np.mean(cluster))

    # 4. Reconstruct with Anchor Snapping
    page_str = ""
    for line in lines:
        line.sort(key=lambda b: b["x"])
        line_text = ""
        prev_char_pos = 0
        
        # Use a localized character-based grid approach per line
        # but anchor the starting x-positions
        for b in line:
            # snap b["x"] to nearest anchor
            best_a = anchors[0]
            min_d = abs(b["x"] - anchors[0])
            for a in anchors:
                if abs(b["x"] - a) < min_d:
                    min_d = abs(b["x"] - a)
                    best_a = a
            
            # Map best_a to a character position
            target_char_pos = int(best_a / space_width_chars)
            
            if target_char_pos > prev_char_pos:
                line_text += " " * (target_char_pos - prev_char_pos)
            elif prev_char_pos > 0:
                line_text += " "
                
            line_text += b["text"]
            prev_char_pos = len(line_text)
            
        page_str += line_text.rstrip() + "\n"
            
    return page_str
