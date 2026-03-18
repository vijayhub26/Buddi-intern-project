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


def _de_clump(text: str) -> str:
    """
    Split words that OCR often glues together incorrectly.
    """
    import re
    # 1. Split [lowercase][Uppercase] (e.g., EffectiveDate -> Effective Date)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # 2. Split [Letter][Digit] and [Digit][Letter] 
    # (e.g., for3months -> for 3 months, GSTofO% -> GST of 0%)
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    
    # 3. Split common glued phrases specific to LinkedIn/Invoice layouts
    GLUED_PHRASES = [
        "Whenyou", "youroffer", "offerends", "endson", "youlpay", "plusapplicable", 
        "applicabletaxes", "eachmonth", "until", "youcancel", "anytimeby", 
        "clickingthe", "thehomepage", "Settings&Privacy", "ManagePremium", 
        "PremiumAccount", "offor", "byclicking", "AsaGoods", "ServicesTax",
        "registerednonresident", "IndiaGSTof", "asrequiredunder", "taxlaws", 
        "Ifyouare", "yoursubscription", "Pleasevisit", "HelpCenter", 
        "WeiFengLow", "Revenue-APAC"
    ]
    for phrase in GLUED_PHRASES:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        if phrase == "Whenyou": text = pattern.sub("When you", text)
        elif phrase == "youroffer": text = pattern.sub("your offer", text)
        elif phrase == "offerends": text = pattern.sub("offer ends", text)
        elif phrase == "endson": text = pattern.sub("ends on", text)
        elif phrase == "youlpay": text = pattern.sub("you pay", text)
        elif phrase == "plusapplicable": text = pattern.sub("plus applicable", text)
        elif phrase == "applicabletaxes": text = pattern.sub("applicable taxes", text)
        elif phrase == "eachmonth": text = pattern.sub("each month", text)
        elif phrase == "youcancel": text = pattern.sub("you cancel", text)
        elif phrase == "anytimeby": text = pattern.sub("anytime by", text)
        elif phrase == "clickingthe": text = pattern.sub("clicking the", text)
        elif phrase == "thehomepage": text = pattern.sub("the homepage", text)
        elif phrase == "Settings&Privacy": text = pattern.sub("Settings & Privacy", text)
        elif phrase == "ManagePremium": text = pattern.sub("Manage Premium", text)
        elif phrase == "PremiumAccount": text = pattern.sub("Premium Account", text)
        elif phrase == "offor": text = pattern.sub("off for", text)
        elif phrase == "byclicking": text = pattern.sub("by clicking", text)
        elif phrase == "AsaGoods": text = pattern.sub("As a Goods", text)
        elif phrase == "ServicesTax": text = pattern.sub("Services Tax", text)
        elif phrase == "registerednonresident": text = pattern.sub("registered nonresident", text)
        elif phrase == "IndiaGSTof": text = pattern.sub("India GST of", text)
        elif phrase == "asrequiredunder": text = pattern.sub("as required under", text)
        elif phrase == "taxlaws": text = pattern.sub("tax laws", text)
        elif phrase == "Ifyouare": text = pattern.sub("If you are", text)
        elif phrase == "yoursubscription": text = pattern.sub("your subscription", text)
        elif phrase == "Pleasevisit": text = pattern.sub("Please visit", text)
        elif phrase == "HelpCenter": text = pattern.sub("Help Center", text)
        elif phrase == "WeiFengLow": text = pattern.sub("Wei Feng Low", text)

    # 4. Split text stuck to punctuation/symbols (e.g., "846.61(plus" -> "846.61 (plus")
    text = re.sub(r'(\d)(\()', r'\1 \2', text)
    text = re.sub(r'(\))([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])(\()', r'\1 \2', text)
    text = re.sub(r'(\d)(\-)', r'\1 \2', text)
    
    return text


def reconstruct_layout(
    ocr_results: OCRResult,
    line_height_tolerance: float = 0.6,
    space_width_chars: float = 10.0,
    page_width: float = None,
    total_columns: int = 120,
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
        
        # De-clump words
        text = _de_clump(text)
        
        x, y = _box_top_left(box)
        right = _box_right(box)
        h = _box_height(box)
        blocks.append({"x": x, "y": y, "right": right, "h": h, "text": text})

    blocks.sort(key=lambda b: (b["y"], b["x"]))

    # --- Header/Footer Filtering ---
    # Define patterns for recurring boilerplate we want to exclude
    EXCLUDE_PATTERNS = [
        r"Tax Invoice from Linkedln",
        r"SAC:?\s?998439",
        r"Linkedln\s?Singapore\s?Pte\s?Ltd",
        r"10\s?Marina\s?Boulevard",
        r"Marina\s?Bay\s?Financial\s?Centre",
        r"Tower\s?2",
        r"Singapore\s?018983",
        r"SG\s?GST:?\s?201109821G",
        r"Page\s*?[\dl]\s*(?:of|lof)\s*\d+",
    ]
    
    filtered_blocks = []
    for block in blocks:
        is_excluded = False
        for pattern in EXCLUDE_PATTERNS:
            if re.search(pattern, block["text"], re.IGNORECASE):
                is_excluded = True
                break
        if not is_excluded:
            filtered_blocks.append(block)
    
    blocks = filtered_blocks

    if not blocks:
        return ""

    # --- Compute average height for line grouping threshold ---
    avg_h = np.mean([b["h"] for b in blocks]) if blocks else 20.0
    threshold = avg_h * line_height_tolerance

    # --- Group blocks into lines (Soft Line Approach) ---
    lines: List[List[dict]] = []
    
    # Blocks are sorted primarily by Y
    for block in blocks:
        placed = False
        # Check existing lines (most recent first) to see if block belongs in the same vertical range
        for line in reversed(lines):
            line_y = np.mean([b["y"] for b in line])
            
            # If the vertical distance is within our relative height threshold,
            # it belongs to this line.
            if abs(block["y"] - line_y) <= threshold:
                line.append(block)
                placed = True
                break
                
        if not placed:
            lines.append([block])

    # --- X-Coordinate Sorting ---
    # Sort all blocks in a "bin" (line) from left to right
    for i in range(len(lines)):
        lines[i] = sorted(lines[i], key=lambda b: b["x"])
        
    # Sort lines vertically by their average Y position (just in case)
    lines.sort(key=lambda line: np.mean([b["y"] for b in line]))

    # --- Assemble each line with Grid-Based Character Anchoring ---
    text_lines: List[str] = []
    
    # Determine page width if not provided
    if page_width is None:
        if blocks:
            page_width = max(b["right"] for b in blocks)
        else:
            page_width = 800.0
            
    # Avoid division by zero
    if page_width <= 0:
        page_width = 800.0

    for line_blocks in lines:
        line_str = ""
        current_col = 0
        prev_block = None
        
        for i, block in enumerate(line_blocks):
            # Calculate expected column index
            col_index = int(round((block["x"] / page_width) * total_columns))
            
            if prev_block is not None:
                # Calculate gap between previous block and current block
                x_gap = block["x"] - prev_block["right"]
                
                # If gap > half an average character width, insert at least 1 space.
                # Otherwise, it's considered part of the same clumped word (no forced space).
                if x_gap > (space_width_chars * 0.5):
                    n_spaces = max(1, col_index - current_col)
                else:
                    n_spaces = max(0, col_index - current_col)
            else:
                # First block on the line, just pad to its column
                n_spaces = max(0, col_index)
                
            line_str += (" " * n_spaces) + block["text"]
            # Update current column position based on current string length
            current_col = len(line_str)
            prev_block = block
            
        text_lines.append(line_str)

    return "\n".join(text_lines)
