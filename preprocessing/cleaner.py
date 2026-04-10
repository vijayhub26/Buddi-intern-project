import cv2
import numpy as np


def to_numpy(image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    import PIL.Image
    if isinstance(image, PIL.Image.Image):
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    raise TypeError(f"Unsupported image type: {type(image)}")


def clean_image(image) -> np.ndarray:
    """
    Preprocessing pipeline for flat JPG invoice scans at 200 DPI.
    1. Gentle denoising  — removes JPG block artifacts without softening strokes
    2. 2× upscale        — gives PaddleOCR more pixels per character
    3. Unsharp mask      — restores stroke crispness lost in bicubic upscale
    4. CLAHE on L only   — boosts contrast without colour artifacts
    5. Grayscale         — OCR input
    """
    img = to_numpy(image)

    # 1. Denoise (h=3 is gentle — keeps thin strokes intact)
    img = cv2.fastNlMeansDenoisingColored(img, None,
                                          h=3, hColor=3,
                                          templateWindowSize=7,
                                          searchWindowSize=21)

    # 2. 2× upscale
    h, w = img.shape[:2]
    img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # 3. Unsharp mask (bicubic softens slightly — this restores edge crispness)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    # 4. CLAHE on L channel only (avoids colour artifacts vs full BGR CLAHE)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 5. Grayscale for OCR
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)