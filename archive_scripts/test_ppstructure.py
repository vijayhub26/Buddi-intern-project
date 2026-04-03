import cv2
import numpy as np
import fitz  # PyMuPDF
import os

from paddleocr import PPStructure, save_structure_res

def main():
    pdf_path = "samples/test_dataset.pdf"
    
    print(f"Loading {pdf_path}...")
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=300)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    print("Initializing PP-Structure...")
    # Initialize PPStructure: English language, hide logs
    table_engine = PPStructure(lang='en', show_log=False, layout=True)
    
    print("Running layout analysis...")
    result = table_engine(img)
    
    print("\n--- PP-STRUCTURE RESULTS ---")
    for region in result:
        print(f"Type: {region['type']}")
        print(f"Box : {region['bbox']}")
        
        # If it's a table or text, print the content
        if 'res' in region and region['res']:
            if region['type'] == 'table':
                print(f"HTML Table:\n{region['res']['html'][:200]}...") # truncate for brevity
            else:
                for line in region['res']:
                    print(f"  {line['text']}")
        print("-" * 40)
        
    save_folder = './results/ppstructure'
    os.makedirs(save_folder, exist_ok=True)
    
    # Save the visualized result
    from PIL import Image
    im_show = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    from copy import deepcopy
    try:
        from paddleocr.ppstructure.utility import draw_structure_result
        font_path = 'doc/fonts/simfang.ttf' # default font path
        # skip drawing if font doesn't exist
    except ImportError:
        pass
        
    save_structure_res(result, save_folder, 'test_dataset')
    print(f"Saved structure results to {save_folder}/test_dataset")

if __name__ == "__main__":
    main()
