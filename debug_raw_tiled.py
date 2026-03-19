import sys
from pipeline.extractor import extract_text_from_pdf
import numpy as np

def debug_raw_tiled():
    print("Extracting raw OCR data at 300 DPI with tiling...")
    # I'll manually call the engine to see what it gets
    from pipeline.ocr_engine import RapidOCREngine
    from pipeline.pdf_renderer import render_pdf_pages
    
    engine = RapidOCREngine()
    pages = list(render_pdf_pages("samples/test_dataset.pdf", dpi=300))
    p2_img = pages[1][1]
    
    results = engine.recognize(p2_img)
    
    print("\n--- RAW RESULTS FOR PAGE 2 (Tiled Pass) ---")
    for box, text, conf in results:
        if "you" in text.lower() or "india" in text.lower() or "gst" in text.lower() or "apply" in text.lower():
            pts = np.array(box, dtype=float)
            y = pts[:, 1].min()
            print(f"Y={y:7.2f} | Conf={conf:.3f} | Text: '{text}'")

if __name__ == "__main__":
    debug_raw_tiled()
