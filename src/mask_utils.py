
import cv2
import numpy as np
from PIL import Image
from colour_detection import detect_colour_annotations

def boxes_to_mask(boxes, image_shape):
  """
  Rasterizes a list of 4-corner boxes (as returned by detect_callipers()
  and get_text_attributes_batch()) into a single binary mask.
  """
  mask = np.zeros(image_shape[:2], dtype=np.uint8)
  for box in boxes:
    pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
  return mask


def build_final_mask(image_path, reader, enhance_fn, calliper_fn, text_fn,
                      colour_fn=detect_colour_annotations, colour_ranges=None,
                      colour_fn_kwargs=None, show_thresh=False):
  """
  Union of calliper detection, text detection and colour-annotation detection.

  NOTE: calliper_fn now returns a mask array directly, considering detect_callipers
  was rewritten to use the edge-map directly rather than shape-matched boxes.

  Args:
    image_path (str): path to the image file.
    reader (easyocr.Reader): shared, pre-instantiated OCR reader.
    enhance_fn, calliper_fn, text_fn, colour_fn: detection functions.
    colour_ranges (dict or None): passed through to colour_fn
    colour_fn_kwargs (dict or None): extra keyword arguments forwarded to colour_fn.
    show_thresh (bool): debug visualisation passthrough.

  Returns:
  np.ndarray: final uint8 binary mask, 255 = remove/inpaint, 0 = leave untouched.
  """
  img = Image.open(image_path).convert('RGB')
  image_shape = np.array(img).shape

  enhanced = enhance_fn(img, show_thresh=False)
  calliper_mask = calliper_fn(enhanced, show_thresh=show_thresh)

  text_boxes, _ = text_fn(image_path, reader, show_thresh=show_thresh)
  text_mask = boxes_to_mask(text_boxes, image_shape)

  # Calliper and text detections are both boxes -> rasterize both into one mask
  box_mask = cv2.bitwise_or(calliper_mask, text_mask)

  colour_fn_kwargs = colour_fn_kwargs or {}
  colour_mask = colour_fn(img, colour_ranges=colour_ranges, **colour_fn_kwargs)

  # Final mask is the union of box-based and colour-based detections
  return cv2.bitwise_or(box_mask, colour_mask)
