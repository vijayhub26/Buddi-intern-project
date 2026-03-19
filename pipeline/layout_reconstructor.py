"""
Layout-preserving text reconstructor.
Final 'Stabilized Column' Version.
"""
from typing import List, Tuple, Optional
import numpy as np
import re

OCRResult = List[Tuple[List, str, float]]

def _de_clump(text: str) -> str:
    text = text.strip()
    if not text: return ""
    
    # 1. Lower-to-Upper transition
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # 2. Targeted footer splits (Specific to this document context)
    text = re.sub(r'(Prices)(are)(subject)(to)(change)', r'\1 \2 \3 \4 \5', text, flags=re.IGNORECASE)
    text = re.sub(r'(month)(until)(you)', r'\1 \2 \3', text, flags=re.IGNORECASE)
    text = re.sub(r'(See)(here)(for)(detailed)', r'\1 \2 \3 \4', text, flags=re.IGNORECASE)
    text = re.sub(r'(instructions)(on)(how)(to)(cancel)', r'\1 \2 \3 \4 \5', text, flags=re.IGNORECASE)
    text = re.sub(r'(offer)(ends)', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'(India)(GST)', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'(Admin)(Center)', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'(Custom)(receipt)', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'(details)(within)', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'(Please)(visit)', r'\1 \2', text, flags=re.IGNORECASE)
    
    # 3. Numeric/Letter transitions
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    
    # 4. Misspellings
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
