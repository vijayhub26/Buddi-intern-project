# Project Analysis Report: Buddi-intern-project (OCR MVP)

This report provides a comprehensive overview of the project's development, the techniques explored, and the current state of the OCR extraction pipeline.

## 1. What is Done / Major Changes
The project has evolved from a basic OCR script into a robust end-to-end pipeline capable of preserving complex document layouts.

- **Engine Migration**: Transitioned to **RapidOCR** (ONNX-based), eliminating the architectural overhead and dependency issues of Tesseract.
- **Layout Fidelity**: Implementation of a **Grid-Based Reconstruction** algorithm that successfully preserves columns, tables, and spacing in plain-text output.
- **Searchable PDF Support**: Built a generation engine ([pdf_writer.py](file:///c:/projects/test/pipeline/pdf_writer.py)) that overlays an invisible, selectable text layer onto the original document images.
- **Smart Post-Processing**:
    - **De-clumping**: Regex-based logic to split words incorrectly joined by the OCR (e.g., `EffectiveDate` → `Effective Date`).
    - **Boilerplate Removal**: Automated filtering of redundant headers/footers (tuned for LinkedIn/Invoice formats).
    - **Symbol Correction**: Fixed common misidentifications like the Rupee symbol (`₹`) being read as `3`.
- **Performance Optimization**: Tuned the preprocessing pipeline to prioritize speed for digital-source PDFs while maintaining high accuracy.

---

## 2. Techniques Tried (Iterative Evolution)
Many techniques were implemented and tested; some were retained, while others were disabled to balance speed and accuracy:

| Technique | Status | Rationale |
| :--- | :--- | :--- |
| **Tesseract OCR** | Discarded | Replaced by RapidOCR for better out-of-the-box accuracy on diverse fonts and zero system dependency. |
| **Linear Extraction** | Discarded | Failed to maintain column structures, resulting in unreadable "collapsed" text. |
| **Denoising (NLMeans)** | Disabled | Removed from the default [clean_image](file:///c:/projects/test/preprocessing/cleaner.py#81-97) flow as it was too slow for digital-first receipts. |
| **Hough Deskewing**| Disabled | Found unnecessary for standard digital PDFs; kept as an optional flag (`--no-deskew`) but off by default. |
| **Hard Binarization** | Disabled | Favored **CLAHE** instead, as binary images often lost detail in small or faint text. |

---

## 3. Techniques Currently Used
The current pipeline uses a "best-of-breed" approach for speed and layout preservation:

### A. Preprocessing ([cleaner.py](file:///c:/projects/test/preprocessing/cleaner.py))
- **Grayscale Conversion**: Essential first step for OCR.
- **CLAHE (Contrast Enhancement)**: Adapts contrast locally to make characters "pop" against noisy or low-contrast backgrounds.

### B. OCR Engine ([ocr_engine.py](file:///c:/projects/test/pipeline/ocr_engine.py))
- **RapidOCR**: Efficient ONNX-based detection and recognition.
- **Confidence Filtering**: Discards low-confidence predictions to reduce noise.

### C. Layout Reconstruction ([layout_reconstructor.py](file:///c:/projects/test/pipeline/layout_reconstructor.py))
- **Soft Line Grouping**: Uses a vertical tolerance (weighted by average character height) to group OCR blocks into visual lines.
- **120-Column Grid Anchoring**: Maps the horizontal position of every word to a fixed grid, ensuring that text in columns aligns perfectly across the entire document.
- **Proportional Space Injection**: Dynamically calculates and inserts spaces based on the physical gaps between bounding boxes.

### D. Reverse Extraction ([pdf_text_extractor.py](file:///c:/projects/test/pipeline/pdf_text_extractor.py))
- **PyMuPDF (Fitz)**: Extracts coordinate-level data from generated PDFs to verify the layout integrity of the searchable layer.

---

## Conclusion
The project is currently in a high-performance state, specifically optimized for **invoice and receipt processing**. The current "Grid + CLAHE" approach represents the peak of the techniques tried, offering a superior balance of layout preservation and processing speed.
