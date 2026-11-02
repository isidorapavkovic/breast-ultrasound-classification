
import re
import cv2
import numpy as np
from PIL import Image

def get_text_attributes_batch(filename: str, reader, show_thresh: bool = False,
                              max_box_fraction: float = 0.25, min_confidence = 0.7):
  """
  Modified version of BUSClean\\'s get_text_attributes (modules/text.py).
  Original: MIT licensed, hawaii-ai/bus-cleaning, commit d2630a4.

  CHANGES from original:
    1. reader is passed in by the caller (created once, outside any loop) instead
       of instantiated fresh inside every cell.
    2. The call to text_helper() and everything it calls is removed entirely.
       text_helper classifies OCR text into clinical semantic categories for BUSClean\\'s
       own knowledge-extraction use case; it plays no role in producing the bounding
       boxes used here for annotation removal.
    3. paragraph=True changed to paragraph=False. Paragraph mode\\'s purpose was
       producing reading-order merged text for semantic classification, which, when
       left as is, merged spatially separate real detections into one bounding box
       spanning tissue between them.
    4. max_box_fraction: rejects any box whose width or height exceeds this fraction
       of the image's shorter dimension, considering the permissive threshold misread
       a textured/hypoechoic tissue region as a large block of text.
    5. min_confidence: paragraph=False restores per-detection confidence scores,
       enabling direct confidence filtering as a more principled backstop against
       the same hallucination issue.

  Args:
    filename (str): path to the image file.
    reader (easyocr.Reader): a single, already-instantiated EasyOCR reader, created
                             once by the caller and reused across calls.
    show_thresh (bool): If True, show prints/images for debugging.
    max_box_fraction (float): reject boxes exceeding this fraction of the image\\'s
                              shorter dimension.
    min_confidence (float): reject detections below this confidence.

  Returns:
    bounding_box_coord (list): 4-corner boxes for each detected text region passing
                               all filters.
    orig_text (str): all raw OCR text found, concatenated, for logging.
  """
  img = Image.open(filename).convert('RGB')
  width, height = img.size
  im = np.array(img)

  for x in range(0, 70):
    im[0] = 0
    appending_black = [im[0]]
    im = np.append(im, appending_black, axis=0)

  orig_text = []
  bounding_box_coord = []
  max_dim = max_box_fraction * min(width, height)

  full_text = reader.readtext(im, x_ths=2.4, paragraph=False, mag_ratio=2.1, low_text=0.2, link_threshold=0.35)

  for bbox, text, confidence in full_text:
    bbox = np.asarray(bbox, dtype='int')
    orig_text.append(text)

    box_w = bbox[2][0] - bbox[0][0]
    box_h = bbox[2][1] - bbox[0][1]

    if (bbox[2][1] > height/8 and len(text) > 2 and re.match(r'^\W+$', " ".join(text.split())) is None
        and box_w <= max_dim and box_h <= max_dim and confidence >= min_confidence):
      if bbox[2][1] > height:
        bbox[2][1] = height
      bounding_box_coord.append([bbox[0], bbox[1], bbox[2], bbox[3]])

      if show_thresh:
        cv2.rectangle(im, bbox[0], bbox[2], (255, 255, 255), 2)

    if show_thresh:
      cv2.circle(im, bbox[0], 5, (255, 255, 255), 2)
      cv2.circle(im, bbox[2], 5, (255, 255, 255), 2)

  orig_text = ' '.join(orig_text)

  if show_thresh:
    display(Image.fromarray(im))

  return bounding_box_coord, orig_text
