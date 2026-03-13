"""
PDF generator using ReportLab.
Converts extracted, layout-preserved text back into a PDF file.
"""
from typing import List, Tuple
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

PageResult = Tuple[int, str]


def _write_text_to_canvas(c: canvas.Canvas, text: str, width: float, height: float):
    """Write multiline text to a ReportLab canvas, aligning to the top-left."""
    textobject = c.beginText()
    
    # Start coordinates (top-left margin)
    margin_x = 40
    margin_y = height - 40
    
    textobject.setTextOrigin(margin_x, margin_y)
    textobject.setFont("Courier", 8)  # Monospace for layout preservation
    
    # Line height
    leading = 10
    textobject.setLeading(leading)

    for line in text.splitlines():
        # Prevent text from drawing completely off the bottom of the page
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
    Generate a new PDF document from extracted text pages.

    Args:
        pages: List of (page_number, text_content) tuples.
        output_pdf_path: Path where the new PDF should be saved.
    """
    from reportlab.lib.pagesizes import landscape, letter
    # Use landscape letter to fit wide tabular data
    page_width, page_height = landscape(letter)
    
    c = canvas.Canvas(output_pdf_path, pagesize=landscape(letter))
    
    for _, page_text in pages:
        _write_text_to_canvas(c, page_text, page_width, page_height)
        c.showPage()  # Finish current page
        
    c.save()
