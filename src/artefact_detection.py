
from itertools import combinations
import cv2
import numpy as np
from PIL import Image, ImageFilter
from modules.artifacts import detect_anno


def enhance_image_full_frame(im: Image.Image, show_thresh: bool = False) -> Image.Image:
  """
  Modified version of BUSClean's enhance_image (modules/artifacts.py).
  Original: MIT licensed, hawaii-ai/bus-cleaning, commit d2630a4.

  CHANGE from original: runs FIND_EDGES on the full, uncropped frame
  instead of blacking out a 15% border: lesions near BUSI's edge were being discarded.

  Args:
    im (Image.Image): RGB PIL image.
    show_thresh (bool): If True, show debugging prints/images.

  Returns:
    Image.Image: the enhanced (edge-map) image.
  """
  dilate = 1
  new_image = im.filter(ImageFilter.FIND_EDGES)
  if show_thresh:
    display(new_image)

  array_im: np.ndarray = np.array(new_image)
  array_im = cv2.cvtColor(array_im, cv2.COLOR_RGB2GRAY)
  _, binary_image = cv2.threshold(array_im, 210, 255, cv2.THRESH_BINARY)
  enhance = (Image.fromarray(binary_image)).filter(ImageFilter.MaxFilter)

  enhance = np.array(enhance)
  for i in range(dilate):
    enhance = cv2.dilate(enhance, np.ones((2, 2), np.uint8))
  enhance = Image.fromarray(enhance)

  if show_thresh:
    display(enhance)

  return enhance


def detect_callipers(im, show_thresh: bool = False,
                     dilation_size: int = 2, min_component_area: int = 22):
  """
  Modified version of BUSClean's detect_anno_BUSI (modules/artifacts.py).
  Original: MIT licensed, hawaii-ai/bus-cleaning, commit d2630a4.

  Treats the enhanced edge-map as the mask directly, dilated to close gaps in
  dashed/dotted lines. Connected-component filtering removes anything below
  min_component_area before dilation. No assumption is made about calliper shape,
  due to high heterogeneity of artefacts in BUSI.

  Args:
    im: the enhanced (edge-map) image, as returned by
        enhance_image_full_frame().
    show_thresh (bool): If True, show debugging prints/images.
    dilation_size (int): closes gaps in dashed/dotted annotation lines.
    min_component_area (int): pixel area below which is a connected component is
                              treated as noise, not a real annotation.

  Returns:
    np.ndarray: binary mask (uint8), same size as im, 255 = flagged.
  """
  arr = np.array(im)

  num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(arr, connectivity=8)
  filtered = np.zeros_like(arr)
  for label_id in range(1, num_labels):
    if stats[label_id, cv2.CC_STAT_AREA] >= min_component_area:
      filtered[labels == label_id] = 255

  kernel = np.ones((dilation_size, dilation_size), np.uint8)
  mask = cv2.dilate(filtered, kernel, iterations=1)

  if show_thresh:
    display(Image.fromarray(mask))
  return mask
