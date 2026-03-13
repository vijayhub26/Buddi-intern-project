import cv2
from pipeline.ocr_engine import RapidOCREngine
from pipeline.layout_reconstructor import reconstruct_layout
from preprocessing.cleaner import clean_image

def test_image(image_path):
    print(f"Loading {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image.")
        return
        
    print("Cleaning image...")
    cleaned = clean_image(img, deskew_enabled=True)
    
    print("Running OCR...")
    engine = RapidOCREngine()
    results = engine.recognize(cleaned)
    
    print(f"Detected {len(results)} text blocks.")
    
    print("\nReconstructing layout...")
    
    # Print the first 10 blocks to see what we actually detected
    print("Sample of detected blocks:")
    for b in results[:10]:
        print(f"Text: '{b[1]}', Box: {b[0]}")
        
    text = reconstruct_layout(results)
    
    print("\nSaving to test_output.pdf...")
    from pipeline.pdf_writer import create_pdf_from_text
    # create_pdf_from_text expects a list of (page_number, text) tuples
    create_pdf_from_text([(1, text)], "test_output.pdf")
    
    print("Saved to test_output.pdf")

if __name__ == "__main__":
    test_image("results/debug_original.png")
