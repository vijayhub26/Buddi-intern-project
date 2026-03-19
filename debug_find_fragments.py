import sys
from pipeline.extractor import extract_text_from_pdf
import numpy as np

def find_fragments():
    print("Extracting raw OCR data...")
    page_results, full_text, pages_data = extract_text_from_pdf(
        "samples/test_dataset.pdf", dpi=200, return_raw=True)
    
    p2_results = pages_data[1]['ocr_results']
    
    print("\n--- LOCATING TARGET FRAGMENTS (Page 2) ---")
    
    for box, text, conf in p2_results:
        low_text = text.lower()
        if "youare" in low_text or "if" in low_text or "oply" in low_text or "only" in low_text or "tax laws" in low_text:
            pts = np.array(box, dtype=float)
            x = pts[:, 0].min()
            y = pts[:, 1].min()
            r = pts[:, 0].max()
            h = pts[:, 1].max() - y
            print(f"Y={y:7.2f} | H={h:4.1f} | X={x:7.2f} | R={r:7.2f} | Text: '{text}'")

if __name__ == "__main__":
    find_fragments()
