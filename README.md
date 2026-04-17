# DocExtract OCR Pipeline

A lightweight, layout-preserving OCR extraction pipeline powered by **PaddleOCR** and **OpenCV**.

## Features
- **GPU-Accelerated**: Fully utilizes local CUDA rendering dynamically to prevent VRAM crashes.
- **Layout Reconstruction**: Intelligently groups and spaces detected text bounding boxes so that the output `.txt` files accurately reflect the spatial column/row layout of the original PDF.
- **High Fidelity Cleaning**: Uses `wordninja` and `pyspellchecker` to eliminate dense token clumping without warping numerical spec sheets.
- **Offline Capable**: Fully portable, relying only on locally downloaded models. 

## 1. Installation

You need Python 3.9 or 3.10 and `pip`.

1. Clone or download this project folder.
2. Initialize a fresh virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the environment:
   - **Windows:** `.\venv\Scripts\activate`
   - **Mac/Linux:** `source venv/bin/activate`
4. Install the locked dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The PaddleOCR models will automatically download on their first execution).*

## 2. Usage

Run the orchestrator script natively from the terminal:

```bash
python run.py --input path/to/your/document.pdf --output results/output.txt
```

### Advanced Arguments
- `--dpi`: Rendering resolution of the PDF (default is 200, use 300+ for dense text).
- `--ignore-symbols`: Filter out dense, noisy non-alphanumeric punctuation.
- `--pages 1 3`: Select specific pages to index natively. 
- `--min-confidence 0.8`: Discards character bounding boxes below 80% OCR certainty.

## 3. Evaluation Toolkit

You can evaluate the OCR against Ground Truth using the provided script:
```bash
python evaluate_performance.py --input samples/batch3-0999.pdf --ground-truth ground_truth/batch3.txt
```
This evaluates the extraction accuracy dynamically, computing metrics for Word Error Rate (WER), Character Error Rate (CER), latency, and layout deltas.
