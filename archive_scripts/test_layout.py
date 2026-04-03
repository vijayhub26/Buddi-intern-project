import cv2
import numpy as np
from pdf2image import convert_from_path
import fitz

doc = fitz.open('samples/test_dataset.pdf')
page = doc.load_page(0)
pix = page.get_pixmap(dpi=300)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
if pix.n == 4:
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
else:
    img = img[:, :, ::-1]

from rapid_layout import RapidLayout
layout_engine = RapidLayout()
res = layout_engine(img)
print(res)
