"""
PDF generator using ReportLab.
Supports two output modes:
  1. create_pdf_from_text   — plain text extraction on a white page (legacy)
  2. create_searchable_pdf  — original image background + invisible OCR text overlay
"""
from typing import List, Tuple, Any, Dict
import io
import numpy as np
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.utils import ImageReader

PageResult = Tuple[int, str]
PageData = Dict[str, Any]


# ---------------------------------------------------------------------------
# Legacy: plain text output on a white background
# ---------------------------------------------------------------------------

def _write_text_to_canvas(c: canvas.Canvas, text: str, width: float, height: float):
    """Write multiline text to a ReportLab canvas, aligning to the top-left."""
    textobject = c.beginText()

    margin_x = 40
    margin_y = height - 40

    textobject.setTextOrigin(margin_x, margin_y)
    textobject.setFont("Courier", 8)

    leading = 10
    textobject.setLeading(leading)

    for line in text.splitlines():
        if textobject.getY() < 40:
            c.drawText(textobject)
            c.showPage()
            textobject = c.beginText()
            textobject.setTextOrigin(margin_x, margin_y)
            textobject.setFont("Courier", 8)
            textobject.setLeading(leading)

        textobject.textLine(line)

    c.drawText(textobject)


def create_pdf_from_text(pages: List[PageResult], output_pdf_path: str):
    """
    Generate a new PDF document from extracted text pages (legacy mode).
    """
    page_width, page_height = landscape(letter)
    c = canvas.Canvas(output_pdf_path, pagesize=landscape(letter))

    for _, page_text in pages:
        _write_text_to_canvas(c, page_text, page_width, page_height)
        c.showPage()

    c.save()


# ---------------------------------------------------------------------------
# Searchable PDF: image background + invisible OCR text overlay
# ---------------------------------------------------------------------------

def _numpy_to_image_reader(img_bgr: np.ndarray) -> ImageReader:
    """Convert a BGR numpy array to a ReportLab ImageReader (via PNG bytes)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    success, buf = cv2.imencode(".png", img_rgb)
    if not success:
        raise RuntimeError("Failed to encode page image as PNG.")
    return ImageReader(io.BytesIO(buf.tobytes()))


def _px_to_pt(px: float, dpi: int) -> float:
    """Convert pixels at a given DPI to PDF points (1 pt = 1/72 inch)."""
    return px * 72.0 / dpi


def create_searchable_pdf(pages_data: List[PageData], output_pdf_path: str):
    """
    Generate a searchable PDF by overlaying invisible OCR text on the
    original page images.

    Each page:
      1. Draw the rendered page image as the background.
      2. Place invisible text blocks using exact OCR bounding box coordinates.

    Args:
        pages_data:       List of dicts from extractor (image, ocr_results, dpi…).
        output_pdf_path:  Destination file path.
    """
    if not pages_data:
        raise ValueError("pages_data is empty — nothing to write.")

    # Use dimensions from the first page to create the canvas
    first = pages_data[0]
    dpi = first["dpi"]
    page_w_pt = _px_to_pt(first["img_w"], dpi)
    page_h_pt = _px_to_pt(first["img_h"], dpi)

    c = canvas.Canvas(output_pdf_path, pagesize=(page_w_pt, page_h_pt))

    for page_data in pages_data:
        img_bgr = page_data["image"]
        ocr_results = page_data["ocr_results"]
        img_h = page_data["img_h"]
        img_w = page_data["img_w"]
        page_dpi = page_data["dpi"]

        # Set page size for this specific page (handles multi-page docs with
        # different sizes, e.g. mixed portrait/landscape)
        pw_pt = _px_to_pt(img_w, page_dpi)
        ph_pt = _px_to_pt(img_h, page_dpi)
        c.setPageSize((pw_pt, ph_pt))

        # --- 1. Draw the original page image as background ---
        img_reader = _numpy_to_image_reader(img_bgr)
        c.drawImage(img_reader, 0, 0, width=pw_pt, height=ph_pt,
                    preserveAspectRatio=False, mask="auto")

        # --- 2. Overlay invisible text ---
        for box, text, confidence in ocr_results:
            if not text.strip():
                continue

            pts = np.array(box, dtype=float)
            x_px    = pts[:, 0].min()
            x_r_px  = pts[:, 0].max()
            y_top_px = pts[:, 1].min()
            y_bot_px = pts[:, 1].max()
            box_h_px = max(y_bot_px - y_top_px, 1)
            box_w_px = max(x_r_px - x_px, 1)

            # Convert to PDF point space (origin = bottom-left → flip Y axis)
            x_pt      = _px_to_pt(x_px, page_dpi)
            box_w_pt  = _px_to_pt(box_w_px, page_dpi)
            box_h_pt  = _px_to_pt(box_h_px, page_dpi)

            # Font size: scale to ~85% of box height (typical cap-height ratio)
            font_size = max(4.0, box_h_pt * 0.85)

            # Y baseline: place it at ~85% down from the top of the PDF box
            # (PDF y=0 is bottom, so we subtract y_top in pixel space then add offset)
            y_top_pt = ph_pt - _px_to_pt(y_top_px, page_dpi)
            # baseline = top of box in PDF coords, then move down by cap-height
            y_baseline_pt = y_top_pt - font_size * 0.85

            # Horizontal scale: stretch text to exactly fill bounding box width
            from reportlab.pdfbase.pdfmetrics import stringWidth as sw
            natural_w = sw(text, "Helvetica", font_size)
            if natural_w > 0:
                h_scale = (box_w_pt / natural_w) * 100.0
            else:
                h_scale = 100.0
            # Clamp scale to avoid extreme distortion for very short strings
            h_scale = max(10.0, min(h_scale, 500.0))

            textobj = c.beginText()
            textobj.setTextRenderMode(3)   # invisible: no fill, no stroke
            textobj.setFont("Helvetica", font_size)
            textobj.setHorizScale(h_scale)
            textobj.setTextOrigin(x_pt, y_baseline_pt)
            textobj.textLine(text)
            c.drawText(textobj)

        c.showPage()

    c.save()
    print(f"  Searchable PDF saved: {output_pdf_path}")
