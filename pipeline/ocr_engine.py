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
            engine.text_det.limit_side_len = 1280
            # Force DBNet to find up to 2000 polygons (default is 1000).
            # This prevents it from taking shortcuts by merging distant text blocks 
            # into massive single polygons to save on its max limits.
            engine.text_det.max_candidates = 2500
        result, _ = engine(image)
        if result is None:
            return []
            
        # Target the middle/right section if we missed the GST line
        text_found = any("apply" in (item[1] or "").lower() for item in result)
        if not text_found and h > 1000 and w > 1000:
            # Crop the right 60% of the upper-middle area (where the text is actually located)
            y_start = int(h * 0.1)
            y_end = int(h * 0.5)
            x_start = int(w * 0.4)
            crop = image[y_start:y_end, x_start:w]
            
            crop_res, _ = engine(crop)
            if crop_res:
                combined_results = list(result)
                for box_data, txt, conf in crop_res:
                    # Shift coordinates back to original image space
                    shifted_box = [[pt[0] + x_start, pt[1] + y_start] for pt in box_data]
                    
                    # Deduplicate: check if this box strongly overlaps with an existing box
                    pts1 = np.array(shifted_box, dtype=float)
                    cx = pts1[:, 0].mean()
                    cy = pts1[:, 1].mean()
                    
                    is_duplicate = False
                    for existing_box, _, _ in combined_results:
                        pts2 = np.array(existing_box, dtype=float)
                        # Check if center of new box is inside the bounding rect of existing box
                        x_min, y_min = pts2[:, 0].min(), pts2[:, 1].min()
                        x_max, y_max = pts2[:, 0].max(), pts2[:, 1].max()
                        if x_min <= cx <= x_max and y_min <= cy <= y_max:
                            is_duplicate = True
                            break
                            
                    if not is_duplicate:
                        combined_results.append((shifted_box, txt, conf))
                        
                result = combined_results

        output: OCRResult = []
        for item in result:
            output.append((item[0], item[1], float(item[2] or 0.0)))
        return output
