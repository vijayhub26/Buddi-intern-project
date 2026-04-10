#!/usr/bin/env python3
"""
Generate project analysis PDF report for the OCR MVP repository.
Writes: reports/project_report.pdf
"""

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import datetime
import os

out_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
out_dir = os.path.normpath(out_dir)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "project_report.pdf")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleLarge', parent=styles['Title'], fontSize=18, leading=22))
styles.add(ParagraphStyle(name='Section', parent=styles['Heading2'], fontSize=12, leading=14, spaceAfter=6))
body = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, leading=12)
mono = ParagraphStyle('Mono', parent=styles['Code'], fontSize=9, leading=11)

# Report content (detailed and precise)
TITLE = "OCR MVP — Project Analysis Report"
GEN_TIME = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

OBJECTIVE = (
    "Primary: Extract text from image-based PDFs while preserving original spatial "
    "layout and produce searchable PDF output. Secondary: provide diagnostics and "
    "quantitative evaluation (WER/CER, clumping, layout-fidelity, latency/memory)."
)

METHODS = (
    "Rendering: pages rendered to BGR numpy arrays using PyMuPDF (pipeline/pdf_renderer.py). "
    "Preprocessing: minimal OpenCV grayscale via preprocessing/cleaner.py. Layout detection: "
    "Surya layout predictor wrapper in pipeline/surya_engine.py. Recognition: PaddleOCR via "
    "pipeline/ocr_engine.py with crop-optimized recognize_crop() and full-page recognize(). "
    "Layout reconstruction uses anchor snapping and table parsing in pipeline/layout_reconstructor.py."
)

TECH = (
    "Python 3.9+, PyMuPDF (fitz), OpenCV, PaddleOCR (paddleocr), Surya (surya-ocr), "
    "ReportLab for PDF output, numpy, wordninja, pyspellchecker, and jiwer for accuracy metrics."
)

PIPELINE = (
    "1) Render pages -> pipeline/pdf_renderer.render_pdf_pages().\n"
    "2) Preprocess images -> preprocessing/cleaner.clean_image() (grayscale).\n"
    "3) Hybrid layout: pipeline/surya_engine.get_surya_engine().analyze_image() -> reading-ordered blocks.\n"
    "4) Recognition: PaddleOCREngine.recognize_crop() per block (hybrid) or recognize() for full page.\n"
    "5) Filter by --min-confidence and exclude patterns.\n"
    "6) Reconstruct layout with pipeline/layout_reconstructor.py: dedupe, line clustering, anchor discovery, "
    "   table parsing (strict grid for Table blocks), _de_clump and _de_fragment post-processing.\n"
    "7) Output: create_searchable_pdf() in pipeline/pdf_writer.py (image background + invisible text overlay)."
)

METRICS = (
    "Transcription: WER & CER via jiwer (evaluate_performance.compute_accuracy()). Default targets: "
    "WER ≤ 5%, CER ≤ 2%.\n"
    "Clumping: dictionary + NLP detection via wordninja and pyspellchecker (compute_clumping_metrics()).\n"
    "Layout fidelity: line counts, avg line length, empty-line ratio, multi-col ratio (compute_layout_metrics()).\n"
    "System: latency (s) and peak memory (MB) via tracemalloc (run_pipeline_timed())."
)

DELIVERABLES = (
    "- CLI: run.py (orchestrates extraction + searchable PDF creation).\n"
    "- Extractor: pipeline/extractor.py (render → preprocess → OCR → reconstruct).\n"
    "- OCR wrappers: pipeline/ocr_engine.py (Paddle) and pipeline/surya_engine.py (Surya layout).\n"
    "- Reconstructor: pipeline/layout_reconstructor.py (anchors, table parser, declump/defrag).\n"
    "- PDF writer: pipeline/pdf_writer.create_searchable_pdf().\n"
    "- Diagnostics: diagnostic_tester.py and evaluate_performance.py for metrics and reports."
)

FINDINGS = (
    "- README mentions rapidocr-onnxruntime and heavy preprocessing (CLAHE/deskew), but current code uses PaddleOCR + Surya and only grayscale preprocessing.\n"
    "- ONNX models exist in models/ but are not wired into runtime (code uses paddleocr). Consider adding an ONNX backend.\n"
    "- Surya prefers GPU; PaddleOCR currently defaults to CPU in pipeline/ocr_engine.py. Ensure device strategy for target machines.\n"
)

LIMITATIONS = (
    "- Memory: pages_data holds full-resolution images for all pages, increasing peak memory for multi-page docs.\n"
    "- Preprocessing is minimal (grayscale). Deskew/CLAHE/denoise are not currently applied.\n"
    "- Tests/CI are missing; no automated benchmarks.\n"
    "- Language support limited: PaddleOCR instantiated with lang='en'."
)

NEXT_STEPS = (
    "Short-term: sync README with code, add deskew/CLAHE flags, expose Paddle GPU toggle.\n"
    "Medium-term: integrate ONNX runtime backend (using models/), stream page output to reduce memory, parallelize safely.\n"
    "Long-term: add unit tests + CI, benchmarking suite, per-page JSON metrics and logging, dashboard.\n"
)

REPRO_TIPS = (
    "Quick commands (run from repo root):\n"
    "  python run.py --input samples/test_dataset.pdf --output results/out.pdf --dpi 300 --hybrid\n"
    "  python diagnostic_tester.py\n"
    "  python evaluate_performance.py --input samples/test_dataset.pdf --ground-truth ground_truth/groundtruth_b1.txt --dpi 300\n"
)

KEY_FILES = (
    "See these files in the repository:\n"
    "  - run.py\n"
    "  - pipeline/extractor.py\n"
    "  - pipeline/ocr_engine.py\n"
    "  - pipeline/surya_engine.py\n"
    "  - pipeline/layout_reconstructor.py\n"
    "  - pipeline/pdf_writer.py\n"
    "  - preprocessing/cleaner.py\n"
    "  - evaluate_performance.py\n"
)

# Build document

doc = SimpleDocTemplate(out_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=50, bottomMargin=50)
flow = []

flow.append(Paragraph(TITLE, styles['Title']))
flow.append(Paragraph(f"Generated: {GEN_TIME}", body))
flow.append(Spacer(1, 12))

flow.append(Paragraph("**Objective**", styles['Section']))
flow.append(Paragraph(OBJECTIVE, body))
flow.append(Spacer(1, 8))

flow.append(Paragraph("**Methods (what we built, precisely)**", styles['Section']))
flow.append(Paragraph(METHODS, body))
flow.append(Spacer(1, 8))

flow.append(Paragraph("**Technology Stack**", styles['Section']))
flow.append(Paragraph(TECH, body))
flow.append(Spacer(1, 8))

flow.append(Paragraph("**Pipeline (step-by-step)**", styles['Section']))
for line in PIPELINE.split('\n'):
    if line.strip():
        flow.append(Paragraph(line.strip(), body))
flow.append(Spacer(1, 8))

flow.append(Paragraph("**Metrics**", styles['Section']))
for p in METRICS.split('\n'):
    if p.strip():
        flow.append(Paragraph(p.strip(), body))
flow.append(Spacer(1, 8))

flow.append(Paragraph("**Deliverables & Artifacts**", styles['Section']))
for p in DELIVERABLES.split('\n'):
    if p.strip():
        flow.append(Paragraph(p.strip(), body))
flow.append(Spacer(1, 8))

flow.append(Paragraph("**Findings & Discrepancies**", styles['Section']))
flow.append(Paragraph(FINDINGS, body))
flow.append(Spacer(1, 8))

flow.append(Paragraph("**Limitations & Risks**", styles['Section']))
flow.append(Paragraph(LIMITATIONS, body))
flow.append(Spacer(1, 8))

flow.append(Paragraph("**Concrete Next Steps / Plan**", styles['Section']))
flow.append(Paragraph(NEXT_STEPS, body))
flow.append(Spacer(1, 8))

flow.append(Paragraph("**Reproduction & Commands**", styles['Section']))
flow.append(Paragraph(REPRO_TIPS.replace('  ', '&nbsp;&nbsp;').replace('\n', '<br/>'), body))
flow.append(Spacer(1, 8))

flow.append(Paragraph("**Key Files**", styles['Section']))
flow.append(Paragraph(KEY_FILES.replace('\n', '<br/>'), body))
flow.append(Spacer(1, 12))

flow.append(Paragraph("End of report.", body))

# Write PDF

doc.build(flow)
print(f"PDF written: {out_path}")
