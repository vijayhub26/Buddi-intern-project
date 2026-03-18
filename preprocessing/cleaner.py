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
    # PIL Image
    import PIL.Image
    if isinstance(image, PIL.Image.Image):
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    raise TypeError(f"Unsupported image type: {type(image)}")


def grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale if not already."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def denoise(image: np.ndarray) -> np.ndarray:
    """Apply non-local means denoising for better OCR accuracy."""
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement on grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def binarize(image: np.ndarray) -> np.ndarray:
    """Adaptive thresholding to produce a clean binary image."""
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )


def deskew(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct skew using Hough line transform.
    Operates on a binary (thresholded) grayscale image.
    """
    # Invert so text pixels are white
    inverted = cv2.bitwise_not(image)
    coords = np.column_stack(np.where(inverted > 0))
    if len(coords) < 5:
        return image  # Not enough points to determine angle

    angle = cv2.minAreaRect(coords)[-1]
    # cv2.minAreaRect returns angle in [-90, 0); normalise to small rotation
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return image  # Skip negligible rotation

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def clean_image(image, deskew_enabled: bool = True) -> np.ndarray:
    """
    Full preprocessing pipeline for an input image.
    Returns a cleaned grayscale numpy array ready for OCR.

    Steps:
        1. Convert to numpy BGR
        2. Grayscale
        3. CLAHE contrast enhancement
        (Denoising and Deskewing removed for digital-first performance)
    """
    img = to_numpy(image)
    img = grayscale(img)
    img = enhance_contrast(img)
    
    return img
