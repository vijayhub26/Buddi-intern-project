"""
PaddleOCR engine wrapper.
Accepts a preprocessed numpy image and returns detected text blocks
with bounding boxes and confidence scores.
"""
from typing import List, Tuple
import numpy as np

# PaddleOCR returns: list of [box, text, score]
OCRResult = List[Tuple[List, str, float]]


class PaddleOCREngine:
    """
    Backend using PaddleOCR. Lazy-loaded on first use.
    """

    def __init__(self):
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from paddleocr import PaddleOCR
            self._engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        return self._engine

    def recognize(self, image: np.ndarray) -> OCRResult:
        engine = self._get_engine()

        if len(image.shape) == 2:
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        result = engine.ocr(image)

        output: OCRResult = []
        if not result or not result[0]:
            return []

        for line in result[0]:
            box = line[0]        # List of [x, y]
            text = line[1][0]
            score = float(line[1][1])

            # Filter absolute noise (tiny boxes < 10px tall)
            h_box = max(p[1] for p in box) - min(p[1] for p in box)
            if h_box < 10:
                continue

            # Round coordinates to reduce jitter in the layout grid
            refined_box = [[round(pt[0]), round(pt[1])] for pt in box]
            output.append((refined_box, text, score))

        return output


def get_ocr_engine() -> PaddleOCREngine:
    """Return the PaddleOCR engine."""
    return PaddleOCREngine()
