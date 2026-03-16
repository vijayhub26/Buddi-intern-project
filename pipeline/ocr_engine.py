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
            # digital-first: text_score=0.6 reduces noise for sharp digital text
            self._engine = RapidOCR(text_score=0.6, print_verbose=False)
            
            # Prevent cropping of high-DPI images (like 300 DPI A4/Letter size)
            # Default is limit_type='min' and limit_side_len=736 which crops wide images.
            if hasattr(self._engine, 'text_det'):
                self._engine.text_det.limit_type = 'max'
                self._engine.text_det.limit_side_len = 3500

        return self._engine

    def recognize(self, image: np.ndarray) -> OCRResult:
        """
        Run OCR on a preprocessed image.
        Dynamically adjusts the engine's dimension limits to the image size
        to ensure full-page processing without dropping right-aligned text.
        """
        engine = self._get_engine()
        h, w = image.shape[:2]
        
        # Dynamically set the limit to the longest edge so it doesn't crop
        max_dim = max(h, w)
        if hasattr(engine, 'text_det'):
            engine.text_det.limit_type = 'max'
            # Limit the max dimension to a reasonable size for DBNet (e.g. 1280)
            # instead of 3500+ which can cause the model to crop or fail on the right side.
            engine.text_det.limit_side_len = 1280
            
        result, _ = engine(image)
        if result is None:
            return []

        output: OCRResult = []
        for item in result:
            output.append((item[0], item[1], float(item[2] or 0.0)))
        return output
