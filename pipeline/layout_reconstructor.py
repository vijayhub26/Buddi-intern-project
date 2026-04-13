"""
Layout-preserving text reconstructor.
Final 'Stabilized Column' Version.
"""
from typing import List, Tuple, Optional
import numpy as np
import re
import wordninja
from spellchecker import SpellChecker

# Initialize global spellchecker once (to avoid loading dictionary on every run)
spell = SpellChecker()
import wordninja

OCRResult = List[Tuple[List, str, float]]

def _de_clump(text: str) -> str:
    """
    Fix OCR word-clumping using structural patterns only.
    No hardcoded vocabulary ΓÇö works on any document.
    """
    text = text.strip()
    if not text:
        return ""

    # 0. Fast-path exclusions for emails and URLs (which are typically in their own OCR box)
    if '@' in text or '.com' in text.lower() or 'www.' in text.lower():
        return text
        
    # Protect proper nouns/brands that use CamelCase
    text = text.replace('LinkedIn', '@@LINKEDIN@@')

    # 1. CamelCase split: 'invoiceDate' -> 'invoice Date'
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # 2. Colon spacing: 'SAC:998439' -> 'SAC: 998439'
    text = re.sub(r'(:)([^\s])', r'\1 \2', text)

    # 2.5 Punctuation spacing: 'Cancel.See' -> 'Cancel. See' (ignores digits like 18,000)
    text = re.sub(r'([.?!,;])([A-Za-z])', r'\1 \2', text)

    # 3. AlphaΓåÆDigit boundary: 'Invoice51109' -> 'Invoice 51109'
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)

    # 4. DigitΓåÆAlpha boundary: '2013Invoice' -> '2013 Invoice'
    text = re.sub(r'(\d)([A-Za-z]{2,})', r'\1 \2', text)

    # 5. Generic Dictionary-based NLP Word Splitter
    #    For any token >= 7 chars, check if it's a valid English word.
    #    If it's not (e.g. 'tochange', 'detailswithin'), use wordninja to split it.
    tokens = text.split()
    new_tokens = []
    
    for tok in tokens:
        clean_tok = tok.strip('.,;:!?()\'"')
        
        # We only consider splitting tokens that are pure letters and >= 7 chars.
        if len(clean_tok) >= 7 and clean_tok.isalpha():
            # Skip splitting TitleCase proper nouns like 'Bharath'
            if clean_tok.istitle():
                new_tokens.append(tok)
                continue
                
            # If the token is NOT a known English word, use NLP to split it
            # (Valid words like 'Invoice', 'instructions', 'Singapore' will be skipped)
            if clean_tok.lower() not in spell:
                split_words = wordninja.split(tok)
                new_tokens.append(" ".join(split_words))
            else:
                new_tokens.append(tok)
        else:
            new_tokens.append(tok)
            
    text = " ".join(new_tokens)

    # 6. OCR misspelling correction for 'I' -> 'l'
    text = re.sub(r'\blf\b', 'If', text)

    # Restore protected brands
    text = text.replace('@@LINKEDIN@@', 'LinkedIn')

    return text


def _de_fragment(text: str) -> str:
    """Fix OCR fragmentation (over-splitting) errors on the fully reconstructed layout."""
    if not text:
        return text

    # 0. Fix common OCR "Page 1 of X" misreadings where '1' is read as 'l' or 'I'
    text = re.sub(r'\bPage\s+[lI1!]\s+of\b', 'Page 1 of', text, flags=re.IGNORECASE)
    text = re.sub(r'\bPagel\s+of\b', 'Page 1 of', text, flags=re.IGNORECASE)

    # 1. Single-lowercase-letter orphans (e.g., 'Bharat h' -> 'Bharath')
    #    Valid 1-letter words: 'A', 'I', 'a', 'i'. Also exclude common punctuation context like "He's" or "don't"
    text = re.sub(r'\b([A-Za-z]{3,})\s+([b-hj-ru-z])\b', r'\1\2', text)

    # 2. Fix ID prefixes: 'P 823194156' -> 'P823194156'
    text = re.sub(r'\bP\s+(\d{6,})\b', r'P\1', text)

    # 3. Clean up spaces around @ and common TLDs
    text = re.sub(r'\s*@\s*', '@', text)
    text = re.sub(r'\s*\.\s*(com|net|org|co|in|edu)\b', r'.\1', text, flags=re.IGNORECASE)

    # 4. Clean fragmented usernames in emails (e.g. 'vijay dm 26@gmail.com')
    #    Match alphanumeric chunks separated by 1 or 2 spaces max before an @
    #    This strict `[ ]{1,2}` prevents eating large layout column gaps!
    def stitch_email(match):
        user = match.group(1)
        domain = match.group(2)
        # Avoid catching standard English phrases that just happen to end in an email
        if any(w in user.lower() for w in [' at ', ' me ', ' contact ', ' reach ', ' email ']):
            return match.group(0)
        return user.replace(' ', '') + '@' + domain

    text = re.sub(r'\b((?:[a-zA-Z0-9._-]+[ ]{1,2}){1,3}[a-zA-Z0-9._-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', stitch_email, text)

    # 5. Collapse '$ ' before digits — OCR returns $ and amount as separate boxes
    #    '$ 564,02' -> '$564,02',  '$ 5 640,17' -> '$5 640,17'
    text = re.sub(r'\$\s+(\d)', r'$\1', text)

    # 6. Strip spurious OCR period after lone row numbers (commented out — valid in GT)
    #    text = re.sub(r'^(\s*\d{1,2})\.\s+(?=[A-Z])', r'\1 ', text, flags=re.MULTILINE)

    # 7. Split price@spec merges — PaddleOCR fuses the gross-worth value with the
    #    next description continuation line when they share the same Y-band.
    #    e.g. '...527,97@3.40 Ghz' -> '...527,97\n<same-indent>@ 3.40 Ghz'
    #    Uses the host line's own leading whitespace so the split line stays
    #    column-aligned instead of dumping at column 0.
    def _split_at_spec(line):
        m = re.search(r'(\d[\d,]+)@(\d)', line)
        if not m:
            return line
        indent = ' ' * (len(line) - len(line.lstrip()))
        return line[:m.end(1)] + '\n' + indent + '@ ' + line[m.end(1) + 1:]
    text = '\n'.join(_split_at_spec(ln) for ln in text.split('\n'))

    # 8. Re-inject missing sequential item numbers in the "No." column.
    #    PaddleOCR's DBNet detector often misses tiny isolated index tokens
    #    (e.g. "1.", "4.", "5.", "6.") in the narrow No. column.
    #    Strategy:
    #      a) Detect the column positions (no_col, desc_col) from lines that DO
    #         carry a visible number inside the ITEMS block.
    #      b) Walk every line; if it has the qty+UM+price signature of a main item
    #         row but no number at no_col, inject the next counter value.
    #    Processed per-[Page] so differing indentation across pages is handled.
    def _renumber_page(page_text: str) -> str:
        lines = page_text.split('\n')

        # -- Step 1: discover column positions from a line that has a visible index --
        no_col = desc_col = None
        in_items = False
        for line in lines:
            s = line.strip()
            if s == 'ITEMS':
                in_items = True
                continue
            if s == 'SUMMARY':
                break
            if not in_items:
                continue
            m = re.match(r'^(\s+)(\d{1,2})\.?\s{3,}(\w)', line)
            if m and re.search(r'\d,\d\d\s{2,}\w', line):
                no_col   = len(m.group(1))   # column where the digit lives
                desc_col = m.start(3)         # column where description begins
                break

        if no_col is None:          # no reference line detected — leave unchanged
            return page_text

        # -- Step 2: re-walk and inject numbers where absent --
        # A "main item row" has:  quantity (X,XX)  ·  unit word  ·  price digits
        item_row_re = re.compile(r'\d,\d\d\s{2,}\w{2,}\s{2,}[\d,]')
        numbered_re = re.compile(r'^\s{' + str(no_col) + r'}(\d{1,2})\.?\s')

        result   = []
        in_items = False
        counter  = 0
        for line in lines:
            s = line.strip()
            if s == 'ITEMS':
                in_items = True
                counter  = 0
            elif s == 'SUMMARY':
                in_items = False

            if in_items and item_row_re.search(line):
                m_n = numbered_re.match(line)
                if m_n:
                    # Line already has an index → sync counter so we stay in step
                    counter = int(m_n.group(1))
                else:
                    # Missing index → inject
                    counter  += 1
                    num       = str(counter)
                    pad_after = ' ' * max(1, desc_col - no_col - len(num))
                    line      = ' ' * no_col + num + pad_after + line[desc_col:]

            result.append(line)
        return '\n'.join(result)

    # Split on [Page N] markers so each page is re-numbered independently
    parts = re.split(r'(\[Page \d+\]\n)', text)
    text  = ''.join(
        _renumber_page(p) if not re.match(r'\[Page \d+\]\n', p) else p
        for p in parts
    )

    return text



def reconstruct_layout(
    ocr_results: OCRResult,
    image: np.ndarray = None,
    line_height_tolerance: float = 0.5,
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
            if gap_ratio > 1.25:
                extra_newlines = max(1, int(round(gap_ratio)) - 1)
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