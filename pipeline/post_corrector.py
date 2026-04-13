"""
post_corrector.py
=================
LLM-based OCR post-correction using a local Ollama model.

Hallucination-prevention strategy:
  1. All numbers, codes, emails, URLs, and date-like tokens are extracted and
     replaced with opaque placeholders (<N0>, <N1>, …) BEFORE the LLM sees the text.
  2. The LLM is instructed to fix ONLY obvious character-level OCR typos.
  3. Placeholders are restored verbatim after the LLM responds.
  4. Temperature is pinned to 0 for deterministic output.

Usage:
    from pipeline.post_corrector import PostCorrector
    corrector = PostCorrector()                       # uses qwen3:8b by default
    clean_text = corrector.correct(raw_ocr_text)
"""

import re
import time
import urllib.request
import urllib.error
import json
from typing import Optional

# ---------------------------------------------------------------------------
# Patterns that must NOT be touched by the LLM
# ---------------------------------------------------------------------------
_PROTECTED_PATTERNS = [
    # Emails
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
    # URLs / IBANs / codes with mixed alphanum+punctuation (e.g. GB75MCRL06841367619257)
    r"[A-Z]{2}\s?\d{2}\s?[A-Z0-9 ]{12,}",
    # Prices / quantities  (e.g. 1 394,67 / 37,75 / 4/8/16)
    r"\d[\d ,./\-]+\d",
    # Standalone numbers
    r"\b\d+\b",
    # Percentage
    r"\d+\s*%",
    # Dates
    r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}",
    # Part codes like GX-212JC, i5-4570
    r"\b[A-Za-z]{1,4}[\-]\d[\w\-]*\b",
    # Units (standalone)
    r"\b(GB|MB|TB|GHz|MHz|mhz|each|UM|VAT|DPI|RAM|SSD|HD|LCD)\b",
]

_PROTECTED_RE = re.compile(
    "|".join(f"(?:{p})" for p in _PROTECTED_PATTERNS)
)


def _mask_protected(text: str) -> tuple[str, dict]:
    """Replace protected tokens with <Nn> placeholders. Returns (masked_text, mapping)."""
    mapping = {}
    counter = [0]

    def replacer(match):
        key = f"<N{counter[0]}>"
        mapping[key] = match.group(0)
        counter[0] += 1
        return key

    masked = _PROTECTED_RE.sub(replacer, text)
    return masked, mapping


def _restore_protected(text: str, mapping: dict) -> str:
    """Substitute placeholders back with their original values."""
    for key, value in mapping.items():
        text = text.replace(key, value)
    return text


# ---------------------------------------------------------------------------
# Ollama client (no extra dependency — uses stdlib urllib)
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"

_SYSTEM_PROMPT = """You are a precise OCR post-correction assistant.

Rules you MUST follow:
1. Fix ONLY clear character-level OCR errors (e.g. 'rn' read as 'm', 'l' as '1', '0' as 'O').
2. NEVER change, add, or remove any <N0>-style placeholder tokens — restore them exactly as-is.
3. NEVER invent, paraphrase, reorder, or expand any content.
4. NEVER fix things you are unsure about — leave them unchanged.
5. NEVER change proper nouns, brand names, or abbreviations unless the OCR error is 100% obvious.
6. Preserve ALL whitespace, newlines, and column alignment exactly.
7. Return ONLY the corrected text. No explanation, no markdown, no commentary."""


def _call_ollama(prompt: str, model: str, timeout: int) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "system": _SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1.0,
            "num_predict": 2048,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return body.get("response", "")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
class PostCorrector:
    """
    OCR post-corrector backed by a local Ollama LLM.

    Parameters
    ----------
    model : str
        Ollama model tag to use. Defaults to 'qwen3:8b' (already on your machine).
        Fallback: 'llama3.2:3b' if qwen3:8b is unavailable.
    chunk_lines : int
        Number of text lines per LLM call. Smaller = less hallucination risk,
        more API calls. Default 20.
    timeout : int
        Per-request timeout in seconds.
    enabled : bool
        Set to False to bypass correction (useful for A/B testing).
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        chunk_lines: int = 20,
        timeout: int = 120,
        enabled: bool = True,
    ):
        self.model = model
        self.chunk_lines = chunk_lines
        self.timeout = timeout
        self.enabled = enabled
        self._available = self._check_ollama()

    def _check_ollama(self) -> bool:
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/tags", method="GET"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                available = [m["name"] for m in data.get("models", [])]
                if self.model not in available:
                    print(
                        f"[PostCorrector] WARN: Model '{self.model}' not found. "
                        f"Available: {available}. Falling back to llama3.2:3b."
                    )
                    # Graceful fallback
                    if "llama3.2:3b" in available:
                        self.model = "llama3.2:3b"
                    else:
                        return False
                print(f"[PostCorrector] OK: Using model: {self.model}")
                return True
        except Exception as e:
            print(f"[PostCorrector] ERROR: Ollama not reachable: {e}. Correction disabled.")
            return False

    def _correct_chunk(self, chunk: str) -> str:
        """Mask → LLM → restore for a single text chunk."""
        masked, mapping = _mask_protected(chunk)

        prompt = (
            "Correct any OCR character errors in the text below. "
            "Do not touch the <N…> placeholder tokens:\n\n"
            f"{masked}"
        )

        try:
            response = _call_ollama(prompt, self.model, self.timeout)
            restored = _restore_protected(response.strip(), mapping)
            return restored
        except Exception as e:
            print(f"[PostCorrector] ✗ LLM call failed: {e}. Returning original chunk.")
            return chunk  # safe fallback: return original

    def correct(self, text: str, verbose: bool = False) -> str:
        """
        Run LLM post-correction on OCR text layout.
        
        Layout Safety Guarantee:
        Splits text by multi-spaces (2+) and newlines. The LLM ONLY sees
        isolated string segments, making it physically impossible for it 
        to collapse formatting grids or delete rows.
        """
        if not self.enabled or not self._available:
            return text

        # Split text into blocks of actual text vs structural layout whitespaces
        segments = re.split(r'(\s{2,}|\n)', text)
        
        corrected_segments = []
        t0 = time.time()
        
        calls = 0
        for seg in segments:
            # Pass purely structural whitespace through completely untouched
            if not seg or seg.isspace():
                corrected_segments.append(seg)
                continue
                
            # For segments with text, protect their single-space padding
            l_space = len(seg) - len(seg.lstrip(' '))
            r_space = len(seg) - len(seg.rstrip(' '))
            core = seg.strip()
            
            if not core:
                corrected_segments.append(seg)
                continue
                
            l_str = " " * l_space
            r_str = " " * r_space
            
            calls += 1
            res = self._correct_chunk(core)
            corrected_segments.append(l_str + res + r_str)

        elapsed = time.time() - t0
        if verbose:
            print(f"[PostCorrector] Processed {calls} text segments in {elapsed:.1f}s")
            
        return "".join(corrected_segments)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = """
    Invoice no: 51l09338

    Date of issue:                              04/13/2013

    Andrews, Kirby and Va1dez                    Becker Ltd
    58861 Gonza1ez Prairie                       8012 Stewart Summit Apt. 455

    IBAN: GB 75 MCRL 06841367619257

    No. Description                   Qty   UM      Net price Net worth   VAT [%]
     1. CLEARANCE! Fast De11 Desktop  3,00  each    209,00    627,00      10%
    """

    corrector = PostCorrector()
    result = corrector.correct(sample, verbose=True)
    print("\n--- CORRECTED ---")
    print(result)
