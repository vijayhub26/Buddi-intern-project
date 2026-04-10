import cv2
import numpy as np
from pipeline.pdf_renderer import render_pdf_pages
from pipeline.extractor import get_ocr_engine
from pipeline.layout_reconstructor import reconstruct_layout
from preprocessing.cleaner import clean_image
import time
import sys

def test_phases(pdf_path: str):
    print("--- PIPELINE PHASE DIAGNOSTICS ---")
    
    # PHASE 1: Rendering & Cleaning
    print("\n[PHASE 1: Rendering & Cleaning]")
    start = time.time()
    pages = list(render_pdf_pages(pdf_path, dpi=200))
    _, rendered_image = pages[0]
    cleaned_image = clean_image(rendered_image)
    print(f"Rendered image shape: {rendered_image.shape}")
    print(f"Cleaned image shape: {cleaned_image.shape}")
    print(f"Phase 1 Time: {time.time() - start:.3f}s")

    # PHASE 2: OCR Extraction
    print("\n[PHASE 2: OCR Extraction (PaddleOCR)]")
    start = time.time()
    engine = get_ocr_engine()
    raw_ocr = engine.recognize(cleaned_image)
    if raw_ocr:
        print(f"Extracted {len(raw_ocr)} bounding boxes.")
        # Find exactly where it hallucinates 846.61
        print("Checking text for '846.61' or '2846.61':")
        for box, txt, score in raw_ocr:
            if '846' in txt or '2846' in txt or '3846' in txt:
                print(f"  -> RAW PaddleOCR reading: '{txt}' (Conf: {score:.3f})")
            if 'Pagel' in txt:
                print(f"  -> RAW PaddleOCR reading: '{txt}' (Conf: {score:.3f})")
    else:
        print("OCR returned nothing.")
    print(f"Phase 2 Time: {time.time() - start:.3f}s")
    
    # PHASE 3: Layout Reconstruction
    print("\n[PHASE 3: Layout Reconstruction]")
    start = time.time()
    img_h, img_w = cleaned_image.shape[:2]
    layout_text = reconstruct_layout(raw_ocr, image=cleaned_image, page_width=img_w)
    print("Extracted first 15 lines of Layout:")
    lines = layout_text.split('\n')[:15]
    for i, l in enumerate(lines):
        print(f"  {i+1}: {l}")
    print(f"Phase 3 Time: {time.time() - start:.3f}s")

if __name__ == "__main__":
    test_phases("samples/test_dataset.pdf")
