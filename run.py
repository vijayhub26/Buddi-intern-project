"""
CLI entry point for the OCR MVP.

Usage:
    python run.py --input path/to/file.pdf
    python run.py --input path/to/file.pdf --output results/output.txt --dpi 300
    python run.py --input path/to/file.pdf --pages 1 3 5   # specific pages only
"""
import argparse
import os
import sys
from datetime import datetime
from pipeline.extractor import extract_text_from_pdf


def parse_args():
    parser = argparse.ArgumentParser(
        description="OCR MVP — Extract text from image-based PDFs preserving layout."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the input PDF file."
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path for the output .txt file. Defaults to results/<pdf_name>_<timestamp>.txt"
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Rendering DPI for PDF pages (default: 300)."
    )
    parser.add_argument(
        "--no-deskew", action="store_true",
        help="Disable automatic skew correction."
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.0,
        help="Minimum OCR confidence to include a text block (0.0–1.0)."
    )
    parser.add_argument(
        "--pages", nargs="+", type=int, default=None,
        help="Space-separated list of page numbers to process (1-indexed). All pages if omitted."
    )
    return parser.parse_args()


def progress(page_num: int, total: int):
    bar_len = 30
    filled = int(bar_len * page_num / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] Page {page_num}/{total}", end="", flush=True)


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output is None:
        os.makedirs("results", exist_ok=True)
        base = os.path.splitext(os.path.basename(args.input))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join("results", f"{base}_{ts}.pdf")

    print(f"\n OCR MVP — RapidOCR + OpenCV")
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print(f"  DPI    : {args.dpi}")
    print(f"  Deskew : {not args.no_deskew}")
    if args.pages:
        print(f"  Pages  : {args.pages}")
    print()

    page_results, _ = extract_text_from_pdf(
        pdf_path=args.input,
        dpi=args.dpi,
        deskew=not args.no_deskew,
        min_confidence=args.min_confidence,
        pages=args.pages,
        progress_callback=progress,
    )
    print()  # newline after progress bar

    # Write output to PDF
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    from pipeline.pdf_writer import create_pdf_from_text
    create_pdf_from_text(page_results, args.output)

    print(f"\n Done! {len(page_results)} page(s) processed.")
    print(f"  Output saved to: {args.output}\n")


if __name__ == "__main__":
    main()
