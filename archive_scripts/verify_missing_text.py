import cv2
import numpy as np
from pipeline.extractor import extract_text_from_pdf

def debug_boxes():
    print("Extracting with RapidOCR...")
    page_results, full_text, pages_data = extract_text_from_pdf(
        "samples/test_dataset.pdf", dpi=200, deskew=True, return_raw=True)
    
    # We only care about page 2 (index 1)
    p2_data = pages_data[1]
    img = p2_data['image'].copy()
    
    for box, text, conf in p2_data['ocr_results']:
        # If it's the target area
        if "apply" in text.lower() or "gst" in text.lower() or "charge" in text.lower():
            pts = np.array(box, dtype=np.int32)
            # Draw green box
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            print(f"Conf: {conf:.3f} | Box: {pts.tolist()} | Text: {text}")
            
    cv2.imwrite("results/debug_boxes_page2.png", img)
    print("Saved bounding box visualization to results/debug_boxes_page2.png")

if __name__ == "__main__":
    debug_boxes()
