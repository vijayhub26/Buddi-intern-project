
from pipeline.extractor import extract_text_from_pdf
import os

def check_ocr_raw():
    pdf_path = "samples/test_dataset.pdf"
    if not os.path.exists(pdf_path):
        print(f"File {pdf_path} not found")
        return

    print(f"Extracting raw OCR from {pdf_path}...")
    page_results, full_text, pages_data = extract_text_from_pdf(
        pdf_path, dpi=200, return_raw=True
    )

    # Focus on Page 2 where the clump "Whenyour..." happened
    p2 = pages_data[1]
    print(f"\n--- Page 2 Raw OCR Blocks ---")
    for box, text, conf in p2['ocr_results']:
        if "when" in text.lower() or "offer" in text.lower() or "promotion" in text.lower():
            print(f"Text: '{text}' | Box: {box}")

if __name__ == "__main__":
    check_ocr_raw()
