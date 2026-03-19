import cv2
import numpy as np
from pipeline.pdf_renderer import render_pdf_pages
from rapidocr_onnxruntime import RapidOCR

def debug_mid_crop():
    print("Rendering page 2 at 300 DPI...")
    pages = list(render_pdf_pages("samples/test_dataset.pdf", dpi=300))
    img = pages[1][1]
    h, w = img.shape[:2]
    
    # Target the middle area where Y was ~681 at 200 DPI
    # (681/200) * 300 = 1021.5
    y_start = int(h * 0.15)
    y_end = int(h * 0.45)
    crop = img[y_start:y_end, :]
    
    print(f"Image Size: {w}x{h}")
    print(f"Crop Area: Y=[{y_start}:{y_end}]")
    
    engine = RapidOCR(text_score=0.2) # Very aggressive to catch everything
    result, _ = engine(crop)
    
    print("\n--- FOCUSED OCR RESULTS (Mid-Section Crop) ---")
    if result:
        for box, text, score in result:
            print(f"Score={score:.3f} | Text: '{text}'")
    else:
        print("No text found in crop.")

if __name__ == "__main__":
    debug_mid_crop()
