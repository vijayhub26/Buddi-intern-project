"""
OpenCV-based image cleaning and preprocessing pipeline.
Applied to each PDF page before passing to the OCR engine.
"""
import cv2
import numpy as np


def to_numpy(image) -> np.ndarray:
    """Accept numpy array or PIL Image, always return numpy BGR array."""
    if isinstance(image, np.ndarray):
        return image
    import PIL.Image
    if isinstance(image, PIL.Image.Image):
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    raise TypeError(f"Unsupported image type: {type(image)}")


def grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale if not already."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement on grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def clean_image(image) -> np.ndarray:
    """
    Full preprocessing pipeline for an input image.
    Returns a cleaned grayscale numpy array ready for OCR.

    Steps:
        1. Convert to numpy BGR
        2. Grayscale
        3. 2× upscale (improves OCR on small/faint text)
        4. CLAHE contrast enhancement
        5. Stroke thickening (erode) to rescue thin characters
    """
    img = to_numpy(image)
    img = grayscale(img)

    # Upscale 2x to improve OCR detection on faint/small text
    h, w = img.shape[:2]
    img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    img = enhance_contrast(img)

    # Stroke thickening — erode expands dark text pixels slightly
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)

    return img
