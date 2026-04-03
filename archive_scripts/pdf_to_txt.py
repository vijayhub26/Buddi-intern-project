"""
pdf_to_txt.py
~~~~~~~~~~~~~
Standalone CLI: extract the layout-preserved text layer from a searchable PDF
and write it to a .txt file.

Usage
-----
    python pdf_to_txt.py --input results/linkedin_invoice_searchable.pdf
    python pdf_to_txt.py --input results/out.pdf --output results/out.txt
    python pdf_to_txt.py --input results/out.pdf --pages 1 3 --space-width 5.0
"""
import argparse
import os
import sys
from datetime import datetime

from pipeline.pdf_text_extractor import extract_text_from_searchable_pdf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract layout-preserved text from a searchable PDF → .txt file."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the searchable PDF file.",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help=(
            "Path for the output .txt file. "
            "Defaults to results/<pdf_name>_<timestamp>.txt"
        ),
    )
    parser.add_argument(
        "--pages", nargs="+", type=int, default=None,
        help="Space-separated list of 1-indexed page numbers to include. "
             "All pages if omitted.",
    )
    parser.add_argument(
        "--space-width", type=float, default=6.0,
        dest="space_width",
        help=(
            "Approximate width of one space character in PDF points (default: 6.0). "
            "Decrease for denser text, increase for wider gaps."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Default output path
    if args.output is None:
        os.makedirs("results", exist_ok=True)
        base = os.path.splitext(os.path.basename(args.input))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join("results", f"{base}_{ts}.txt")

    print(f"\n PDF → TXT Extractor")
    print(f"  Input       : {args.input}")
    print(f"  Output      : {args.output}")
    print(f"  Space width : {args.space_width} pt")
    if args.pages:
        print(f"  Pages       : {args.pages}")
    print()

    text = extract_text_from_searchable_pdf(
        pdf_path=args.input,
        pages=args.pages,
        space_width_pts=args.space_width,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)

    lines = text.count("\n") + 1
    print(f"  Done! {lines} lines written.")
    print(f"  Output saved to: {args.output}\n")


if __name__ == "__main__":
    main()
