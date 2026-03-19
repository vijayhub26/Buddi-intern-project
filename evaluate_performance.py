"""
evaluate_performance.py
~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive, document-agnostic performance evaluation for the OCR pipeline.

Measures:
  1. System Performance    — Latency (s), Peak Memory (MB)
  2. Transcription Accuracy — CER, WER (via jiwer)  [requires --ground-truth]
  3. Layout Fidelity        — Line count delta, Column integrity,
                              Avg line length delta, Empty-line ratio

Usage
-----
    python evaluate_performance.py --input <PDF_PATH> --ground-truth <GT_TXT_PATH>
"""

import argparse
import os
import re
import sys
import time
import tracemalloc
import math
from typing import Optional, List


# ──────────────────────────────────────────────────────────────
# 1. Text Normalisation
# ──────────────────────────────────────────────────────────────
def _normalise(text: str) -> str:
    """Lowercase + collapse whitespace for fair comparison."""
    return re.sub(r"\s+", " ", text.lower()).strip()


# ──────────────────────────────────────────────────────────────
# 2. Transcription Accuracy — CER & WER via jiwer
# ──────────────────────────────────────────────────────────────
def compute_accuracy(ground_truth: str, hypothesis: str) -> dict:
    """Return CER and WER using jiwer."""
    try:
        import jiwer
        # Use simple calls for jiwer 3.0+
        wer = jiwer.wer(ground_truth, hypothesis)
        cer = jiwer.cer(ground_truth, hypothesis)
        return {"wer": wer, "cer": cer, "error": None}
    except ImportError:
        return {"wer": None, "cer": None,
                "error": "jiwer not installed. Run: pip install jiwer"}
    except Exception as e:
        return {"wer": None, "cer": None, "error": str(e)}


# ──────────────────────────────────────────────────────────────
# 3. Layout Fidelity — Generic metrics
# ──────────────────────────────────────────────────────────────
def _non_empty_lines(text: str) -> List[str]:
    return [ln for ln in text.splitlines() if ln.strip()]


def compute_layout_metrics(
    hypothesis: str,
    ground_truth: Optional[str] = None,
    custom_fragments: Optional[List[str]] = None,
) -> dict:
    hyp_lines = _non_empty_lines(hypothesis)
    total_hyp_lines = len(hyp_lines)

    # Empty-line Ratio
    all_hyp_lines = hypothesis.splitlines()
    empty_count = sum(1 for ln in all_hyp_lines if not ln.strip())
    empty_ratio = empty_count / len(all_hyp_lines) if all_hyp_lines else 0.0

    # Average line length
    avg_line_len = sum(len(ln) for ln in hyp_lines) / total_hyp_lines if total_hyp_lines else 0.0

    # Column gap detection
    multi_col_lines = sum(1 for ln in hyp_lines if "    " in ln)
    multi_col_ratio = multi_col_lines / total_hyp_lines if total_hyp_lines else 0.0

    # Structural comparison
    gt_metrics = {}
    if ground_truth:
        gt_lines = _non_empty_lines(ground_truth)
        total_gt_lines = len(gt_lines)
        line_delta = total_hyp_lines - total_gt_lines
        avg_gt_len = sum(len(ln) for ln in gt_lines) / total_gt_lines if total_gt_lines else 0.0
        line_len_delta = avg_line_len - avg_gt_len

        gt_metrics = {
            "line_count_delta": line_delta,
            "avg_line_len_delta": line_len_delta,
            "gt_line_count": total_gt_lines,
        }

    # Custom runtime fragment checks
    fragment_results = {}
    if custom_fragments:
        content_flat = _normalise(hypothesis.replace("\n", " "))
        for fragment in custom_fragments:
            norm_frag = _normalise(fragment)
            found = norm_frag in content_flat
            fragment_results[fragment] = "FOUND" if found else "MISSING"

    return {
        "total_lines":       total_hyp_lines,
        "empty_ratio":       empty_ratio,
        "avg_line_len":      avg_line_len,
        "multi_col_ratio":   multi_col_ratio,
        **gt_metrics,
        "custom_fragments":  fragment_results,
    }


# ──────────────────────────────────────────────────────────────
# 4. System Performance
# ──────────────────────────────────────────────────────────────
def run_pipeline_timed(
    pdf_path: str,
    dpi: int,
    min_confidence: float,
    pages: Optional[List[int]],
    exclude_patterns: Optional[List[str]],
) -> dict:
    from pipeline.extractor import extract_text_from_pdf
    tracemalloc.start()
    t0 = time.time()
    _, full_text, _ = extract_text_from_pdf(
        pdf_path=pdf_path, dpi=dpi, deskew=True,
        min_confidence=min_confidence, pages=pages,
        return_raw=True, exclude_patterns=exclude_patterns,
    )
    elapsed = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "full_text":   full_text,
        "latency_s":   elapsed,
        "peak_mem_mb": peak / (1024 ** 2),
    }


# ──────────────────────────────────────────────────────────────
# 5. Report Printer
# ──────────────────────────────────────────────────────────────
def _bar(value: float, total: float = 1.0, width: int = 20) -> str:
    filled = int(round(width * (value / total))) if total else 0
    filled = max(0, min(filled, width))
    return "█" * filled + "░" * (width - filled)


def print_report(perf: dict, accuracy: dict, layout: dict, targets: dict):
    SEP = "═" * 62
    print(f"\n{SEP}")
    print(" OCR PIPELINE — PERFORMANCE EVALUATION REPORT")
    print(SEP)

    # System Performance
    print("\n┌─ SYSTEM PERFORMANCE ──────────────────────────────────────┐")
    print(f"│  Latency         : {perf['latency_s']:.2f} s")
    print(f"│  Peak Memory     : {perf['peak_mem_mb']:.1f} MB")
    print("└───────────────────────────────────────────────────────────┘")

    # Transcription Accuracy
    print("\n┌─ TRANSCRIPTION ACCURACY (jiwer) ──────────────────────────┐")
    if accuracy.get("error"):
        print(f"│  ℹ  {accuracy['error']}")
    else:
        wer, cer = accuracy["wer"], accuracy["cer"]
        wer_pct, cer_pct = wer * 100, cer * 100
        wer_t, cer_t = targets["wer"], targets["cer"]
        print(f"│  WER  {_bar(wer)}  {wer_pct:5.2f}%  {'✅' if wer_pct <= wer_t else '❌'}  (target ≤ {wer_t}%)")
        print(f"│  CER  {_bar(cer)}  {cer_pct:5.2f}%  {'✅' if cer_pct <= cer_t else '❌'}  (target ≤ {cer_t}%)")
    print("└───────────────────────────────────────────────────────────┘")

    # Layout Fidelity
    print("\n┌─ LAYOUT FIDELITY ─────────────────────────────────────────┐")
    print(f"│  Total Lines     : {layout['total_lines']}")
    print(f"│  Avg Line Length : {layout['avg_line_len']:.1f} chars")
    print(f"│  Empty-line Ratio: {layout['empty_ratio'] * 100:.1f}%")
    mc_pct = layout["multi_col_ratio"] * 100
    mc_t = targets["multi_col"]
    mc_ok = "✅" if mc_pct >= mc_t else "⚠️ "
    print(f"│  Multi-col Lines : {mc_pct:.1f}%  {mc_ok}  (target ≥ {mc_t}%)")

    if "line_count_delta" in layout:
        delta = layout["line_count_delta"]
        delta_t = targets["line_delta"]
        delta_ok = "✅" if abs(delta) <= delta_t else "⚠️ "
        sign = "+" if delta >= 0 else ""
        print(f"│  Line Δ vs GT    : {sign}{delta} lines  {delta_ok}  (GT lines: {layout['gt_line_count']}, target Δ ≤ {delta_t})")

    # Custom fragment results
    if layout.get("custom_fragments"):
        print("│")
        print("│  Custom Fragment Checks:")
        for frag, status in layout["custom_fragments"].items():
            icon = "✅" if status == "FOUND" else "❌"
            print(f"│    {icon}  \"{frag}\" → {status}")

    print("└───────────────────────────────────────────────────────────┘")

    # Overall Verdict
    print(f"\n{SEP}")
    if accuracy.get("error") is None and accuracy.get("wer") is not None:
        ok = (accuracy["wer"] * 100 <= targets["wer"] and 
              accuracy["cer"] * 100 <= targets["cer"] and
              (abs(layout.get("line_count_delta", 0)) <= targets["line_delta"]))
        verdict = "✅  PRODUCTION READY" if ok else "⚠️   NEEDS IMPROVEMENT"
    elif "line_count_delta" in layout and abs(layout["line_count_delta"]) <= targets["line_delta"]:
        verdict = "✅  LAYOUT STABLE (no accuracy GT provided)"
    else:
        verdict = "ℹ️   PARTIAL EVALUATION"
    print(f" VERDICT: {verdict}")
    print(f"{SEP}\n")


# ──────────────────────────────────────────────────────────────
# 6. CLI
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Document-agnostic OCR pipeline evaluator.")
    p.add_argument("--input", "-i", required=True, help="Input PDF.")
    p.add_argument("--ground-truth", "-g", default=None, help="Reference .txt.")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--pages", nargs="+", type=int, default=None)
    p.add_argument("--exclude", "-e", nargs="+", default=None)
    p.add_argument("--check-fragments", nargs="+", default=None)
    
    # Target Thresholds
    p.add_argument("--target-wer", type=float, default=5.0, help="Target Max WER % (default: 5.0)")
    p.add_argument("--target-cer", type=float, default=2.0, help="Target Max CER % (default: 2.0)")
    p.add_argument("--target-line-delta", type=int, default=5, help="Max line count difference (default: 5)")
    p.add_argument("--target-multi-col", type=float, default=10.0, help="Min % of lines with gaps (default: 10.0)")
    
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.input):
        print(f"[ERROR] Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"\n OCR Evaluation | Input: {os.path.basename(args.input)}")
    
    result = run_pipeline_timed(
        pdf_path=args.input, dpi=args.dpi, min_confidence=args.min_confidence,
        pages=args.pages, exclude_patterns=args.exclude,
    )
    
    gt_text = None
    if args.ground_truth and os.path.isfile(args.ground_truth):
        with open(args.ground_truth, "r", encoding="utf-8") as f:
            gt_text = f.read()

    accuracy = compute_accuracy(gt_text, result["full_text"]) if gt_text else {"error": "No ground truth."}
    layout = compute_layout_metrics(result["full_text"], gt_text, args.check_fragments)
    
    targets = {
        "wer": args.target_wer, "cer": args.target_cer,
        "line_delta": args.target_line_delta, "multi_col": args.target_multi_col
    }
    print_report(result, accuracy, layout, targets)


if __name__ == "__main__":
    main()
