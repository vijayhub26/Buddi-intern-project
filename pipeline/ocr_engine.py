"""
RapidOCR engine wrapper.
Accepts a preprocessed numpy image and returns detected text blocks
with bounding boxes and confidence scores.
"""
from typing import List, Tuple
import numpy as np

# RapidOCR returns: list of [box, text, score] or None
OCRResult = List[Tuple[List, str, float]]


class RapidOCREngine:
    """
    Thin wrapper around RapidOCR for use in the extraction pipeline.

    Attributes:
        _engine: Lazy-loaded RapidOCR instance (initialised on first call).
    """

    def __init__(self):
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR
            # digital-first: lower text_score to capture smaller/fainter fragments
            self._engine = RapidOCR(text_score=0.2, print_verbose=False)
            
            # Prevent cropping of high-DPI images (like 300 DPI A4/Letter size)
            if hasattr(self._engine, 'text_det'):
                self._engine.text_det.limit_type = 'max'
                self._engine.text_det.limit_side_len = 2560
                self._engine.text_det.max_candidates = 3000

        return self._engine

    def recognize(self, image: np.ndarray) -> OCRResult:
        """
        Run OCR on a preprocessed image using multi-segment tiling for high-res.
        (Restored to fix missing words problem)
        """
        engine = self._get_engine()
        h, w = image.shape[:2]
        
        # Tiling strategy for large images to ensure detection of small fragments
        if h > 1500:
            segment_h = 1600
            overlap = 600
            step = segment_h - overlap
            
            starts = []
            curr = 0
            while curr < h:
                starts.append(curr)
                if curr + segment_h >= h:
                    break
                curr += step
            
            results = []
            for start in starts:
                end = min(start + segment_h, h)
                tile = image[start:end, :]
                tile_res, _ = engine(tile)
                if tile_res:
                    for box_data, txt, conf in tile_res:
                        shifted_box = [[pt[0], pt[1] + start] for pt in box_data]
                        results.append((shifted_box, txt, conf))
        else:
            results, _ = engine(image)
            results = results or []

        output: OCRResult = []
        for item in results:
            output.append((item[0], item[1], float(item[2] or 0.0)))
        return output
