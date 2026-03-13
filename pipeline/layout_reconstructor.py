"""
Layout-preserving text reconstructor.

Given a list of (bounding_box, text, confidence) OCR results from a single
page, reconstructs text that mirrors the original spatial layout:
- Text blocks are sorted by Y (row) then X (column).
- Blocks on the same visual line are joined with spaces proportional to
  horizontal gaps between them.
- Lines are joined with newlines.
"""
from typing import List, Tuple
import numpy as np

OCRResult = List[Tuple[List, str, float]]


def _box_top_left(box) -> Tuple[float, float]:
    """Return (x, y) of the top-left corner of a bounding box."""
    pts = np.array(box, dtype=float)
    x = pts[:, 0].min()
    y = pts[:, 1].min()
    return x, y


def _box_right(box) -> float:
    """Return the rightmost X coordinate of a bounding box."""
    pts = np.array(box, dtype=float)
    return pts[:, 0].max()


def _box_height(box) -> float:
    """Return the approximate height of a bounding box."""
    pts = np.array(box, dtype=float)
    return pts[:, 1].max() - pts[:, 1].min()


def reconstruct_layout(
    ocr_results: OCRResult,
    line_height_tolerance: float = 0.6,
    space_width_chars: float = 18.0,
) -> str:
    """
    Reconstruct text from OCR results preserving the original layout.

    Args:
        ocr_results: List of (box, text, confidence) from the OCR engine.
        line_height_tolerance: Fraction of average box height to use as
            the Y-distance threshold for grouping blocks into the same line.
        space_width_chars: Average character width in pixels (used to decide
            how many spaces to insert between horizontally adjacent blocks).

    Returns:
        A multi-line string that mirrors the spatial layout.
    """
    if not ocr_results:
        return ""

    blocks = []
    import re
    for box, text, score in ocr_results:
        # Common OCR fixes
        # Fix Rupee symbol misread as '3' at the start of a number block
        text = re.sub(r'^3(\d+\.\d{2})', r'₹\1', text)
        text = re.sub(r'^-3(\d+\.\d{2})', r'-₹\1', text)
        
        # Add space between CamelCase words (e.g. EffectiveDate -> Effective Date)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        x, y = _box_top_left(box)
        right = _box_right(box)
        h = _box_height(box)
        blocks.append({"x": x, "y": y, "right": right, "h": h, "text": text})

    blocks.sort(key=lambda b: (b["y"], b["x"]))

    # --- Compute average height for line grouping threshold ---
    avg_h = np.mean([b["h"] for b in blocks]) if blocks else 20.0
    threshold = avg_h * line_height_tolerance

    # --- Group blocks into lines ---
    # We group blocks into the same line if their vertical extents overlap significantly
    lines: List[List[dict]] = []
    
    for block in blocks:
        placed = False
        for line in lines:
            # Check overlap with the first block of the line to see if it belongs here
            line_top = min(b["y"] for b in line)
            line_bottom = max(b["y"] + b["h"] for b in line)
            
            block_top = block["y"]
            block_bottom = block["y"] + block["h"]
            
            # Calculate vertical overlap
            overlap = max(0.0, min(line_bottom, block_bottom) - max(line_top, block_top))
            min_height = min((line_bottom - line_top), block["h"])
            
            if min_height > 0 and (overlap / min_height) > 0.4:
                # Overlaps significantly, add to this line
                line.append(block)
                placed = True
                break
                
        if not placed:
            # Create a new line
            lines.append([block])

    # Sort each line horizontally
    for i in range(len(lines)):
        lines[i] = sorted(lines[i], key=lambda b: b["x"])
        
    # Sort lines vertically by their average Y position
    lines.sort(key=lambda line: np.mean([b["y"] for b in line]))

    # --- Assemble each line with gap-proportional spaces ---
    text_lines: List[str] = []
    for line_blocks in lines:
        line_str = line_blocks[0]["text"]
        for i in range(1, len(line_blocks)):
            prev = line_blocks[i - 1]
            curr = line_blocks[i]
            
            # Distance from end of previous block to start of current block
            gap_px = max(0.0, curr["x"] - prev["right"])
            
            # Depending on DPI, we need a reasonable space count.
            n_spaces = max(1, int(round(gap_px / space_width_chars)))
            line_str += " " * n_spaces + curr["text"]
            
        text_lines.append(line_str)

    return "\n".join(text_lines)
