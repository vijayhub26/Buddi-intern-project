import sys
from pipeline.extractor import extract_text_from_pdf
import numpy as np

def inspect_page_footer():
    page_results, full_text, pages_data = extract_text_from_pdf(
        "samples/test_dataset.pdf", dpi=200, return_raw=True)
    
    p2_results = pages_data[1]['ocr_results']
    
    print("\n--- FULL BLOCK DATA: Page 2 Footer (Y > 600) ---")
    
    blocks = []
    for box, text, conf in p2_results:
        pts = np.array(box, dtype=float)
        y = pts[:, 1].min()
        if y > 600:
            blocks.append((y, box, text, conf))

    # Sort by Y
    blocks.sort(key=lambda x: x[0])
    
    for y, box, text, conf in blocks:
        print(f"Y={y:.2f} | Conf={conf:.3f} | Text: '{text}'")
        print(f"  Box: {box}")

if __name__ == "__main__":
    inspect_page_footer()
