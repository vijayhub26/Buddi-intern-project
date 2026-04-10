import cv2
import numpy as np
import os
from pipeline.pdf_renderer import render_pdf_pages
from pipeline.extractor import get_ocr_engine
from pipeline.layout_reconstructor import _de_clump, _de_fragment
from preprocessing.cleaner import clean_image
import json

def test_pipeline_phases(pdf_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting diagnostics...")
    
    # 1. Page Rendering
    pages = list(render_pdf_pages(pdf_path, dpi=200))
    page_num, rendered_image = pages[0]
    
    # 2. Cleaning
    cleaned_image = clean_image(rendered_image)
    cv2.imwrite(os.path.join(output_dir, "cleaned_page_1.png"), cleaned_image)
    
    # 3. OCR Engine
    engine = get_ocr_engine()
    raw_ocr = engine.recognize(cleaned_image)
    
    # Log raw OCR
    with open(os.path.join(output_dir, "01_raw_ocr.txt"), "w", encoding="utf-8") as f:
        for box, txt, score in sorted(raw_ocr, key=lambda x: x[0][1]): # sort by y
            f.write(f"TXT: '{txt}' | CONF: {score:.3f} | BOX: {box}\n")
            
    # 4. De-clumping Phase
    with open(os.path.join(output_dir, "02_declump_diff.txt"), "w", encoding="utf-8") as f:
        for box, txt, score in raw_ocr:
            declumped = _de_clump(txt)
            if declumped != txt:
                f.write(f"ORIGINAL: '{txt}'  ->  DECLUMPED: '{declumped}'\n")

    # 5. Deduplication & Sorting block
    blocks = []
    for box, txt, score in sorted(raw_ocr, key=lambda x: x[2], reverse=True):
        text = _de_clump(txt)
        if not text: continue
        pts = np.array(box, dtype=float)
        x, y = pts[:, 0].min(), pts[:, 1].min()
        right, bottom = pts[:, 0].max(), pts[:, 1].max()
        h = bottom - y
        cx, cy = (x + right)/2, (y + bottom)/2
        
        is_dup = False
        for b in blocks:
            if abs(cx - b["cx"]) < 10 and abs(cy - b["cy"]) < 10:
                is_dup = True
                if len(text) > len(b["text"]): b["text"] = text
                break
        if not is_dup:
            blocks.append({"x": x, "y": y, "right": right, "h": h, "cx": cx, "cy": cy, "text": text})

    blocks.sort(key=lambda b: (b["y"], b["x"]))
    
    with open(os.path.join(output_dir, "03_blocks.txt"), "w", encoding="utf-8") as f:
        for b in blocks:
            f.write(f"Y: {b['y']:.1f}, X: {b['x']:.1f}, H: {b['h']:.1f} | TEXT: '{b['text']}'\n")

    # 6. Extract Lines
    lines = []
    if blocks:
        current_line = [blocks[0]]
        for b in blocks[1:]:
            if abs(b["y"] - current_line[0]["y"]) < current_line[0]["h"] * 0.6:
                current_line.append(b)
            else:
                lines.append(current_line)
                current_line = [b]
        lines.append(current_line)
        
    with open(os.path.join(output_dir, "04_lines.txt"), "w", encoding="utf-8") as f:
        for i, line in enumerate(lines):
            f.write(f"--- LINE {i} ---\n")
            for b in line:
               f.write(f"  [{b['x']:.1f}] '{b['text']}'\n")

    # 7. Anchoring Mapping
    all_xs = sorted([b["x"] for b in blocks])
    anchors = []
    if all_xs:
        curr_a = all_xs[0]
        cluster = [all_xs[0]]
        for x in all_xs[1:]:
            if x - curr_a < 35:
                cluster.append(x)
            else:
                anchors.append(np.mean(cluster))
                curr_a = x
                cluster = [x]
        anchors.append(np.mean(cluster))
    
    with open(os.path.join(output_dir, "05_anchors.txt"), "w", encoding="utf-8") as f:
        f.write(f"Total blocks X anchors discovered: {len(anchors)}\n")
        f.write(f"{anchors}\n")
        
    # 8. Reconstruct and Defragment
    page_str = ""
    for line in lines:
        line.sort(key=lambda b: b["x"])
        line_text = ""
        prev_char_pos = 0
        
        for b in line:
            best_a = anchors[0]
            min_d = abs(b["x"] - anchors[0])
            for a in anchors:
                if abs(b["x"] - a) < min_d:
                    min_d = abs(b["x"] - a)
                    best_a = a
            
            target_char_pos = int(best_a / 18.0)
            
            if target_char_pos > prev_char_pos:
                line_text += " " * (target_char_pos - prev_char_pos)
            elif prev_char_pos > 0:
                line_text += " "
                
            line_text += b["text"]
            prev_char_pos = len(line_text)
            
        page_str += line_text.rstrip() + "\n"
        
    defrag_str = _de_fragment(page_str)
    
    with open(os.path.join(output_dir, "06_defrag_diff.txt"), "w", encoding="utf-8") as f:
        orig_lines = page_str.splitlines()
        defrag_lines = defrag_str.splitlines()
        for o, d in zip(orig_lines, defrag_lines):
            if o != d:
                f.write(f"ORIGINAL:\n{o}\nDEFRAGGED:\n{d}\n\n")

    with open(os.path.join(output_dir, "07_final_page1.txt"), "w", encoding="utf-8") as f:
        f.write(defrag_str)

    print("Diagnostics saved to", output_dir)

if __name__ == "__main__":
    test_pipeline_phases("samples/test_dataset.pdf", "diagnostics")
