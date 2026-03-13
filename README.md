# OCR MVP: PDF Image-to-Text with Format Preservation

Extract text from image-based PDFs using **RapidOCR** for recognition and **OpenCV** for preprocessing, preserving the original spatial layout, and saving the output as a searchable PDF.

## Features

- **High-Accuracy OCR**: Uses `rapidocr-onnxruntime` for robust multi-language text detection and recognition.
- **Dynamic Resolution Handling**: Automatically adjusts OCR bounds to handle high-resolution (300 DPI) full-page scans without cropping.
- **Layout Preservation**: Custom algorithm reconstructs text by analyzing bounding box coordinates, injecting proportional spacing to recreate visual columns and tables.
- **PDF Generation**: Outputs extracted text into a clean, searchable PDF file using `reportlab`.
- **OpenCV Filtering**: Prepares images with grayscale conversion, denoising, contrast enhancement (CLAHE), and automatic Hough-line deskewing.

## Prerequisites

- Python 3.9+
- Tesseract is *not* required (RapidOCR is fully self-contained ONNX models).

## Installation

```bash
git clone <repository-url>
cd <repository-dir>

# Strongly recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Extract text from a PDF document and save it as a new PDF:

```bash
python run.py --input path/to/document.pdf
```

### Advanced Options

```bash
# Specify an output filename explicitly
python run.py --input document.pdf --output results/my_extracted_data.pdf

# Change the internal rendering resolution (Higher = better OCR, slower speed. Default: 300)
python run.py --input document.pdf --dpi 300

# Disable automatic deskewing (useful if the document has intentional angles)
python run.py --input document.pdf --no-deskew

# Filter out low-confidence OCR predictions (0.0 to 1.0)
python run.py --input document.pdf --min-confidence 0.6

# Process only specific pages (1-indexed)
python run.py --input document.pdf --pages 1 3 5
```

## Examples

If you have a document with two columns of text separated by a large gap (like an invoice):

**Input PDF Image:**
```
Effective Date                                      Transaction ID    
2/27/2026                                           P823194156  
```

Our layout reconstructor preserves the gap dynamically. Standard linear OCR tools would collapse this to:
```
Effective Date Transaction ID
2/27/2026 P823194156
```

## Output

The script natively outputs text back into a `.pdf` file located in the `results/` folder by default. This PDF is rendered using a monospace font ensuring perfect character alignment reflecting the spatial gaps calculated during extraction.
