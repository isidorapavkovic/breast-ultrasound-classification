
import cv2
import numpy as np
from PIL import Image

DEFAULT_COLOUR_RANGES = {
  'green_annotation': ((33, 45, 65), (92, 255, 255)),
  'yellow_annotation': ((16, 50, 70), (38, 255, 255)),
  'diagram_blue_purple': ((90, 40, 70), (162, 255, 255)),
}

DEFAULT_KERNEL_SIZES = {
  'green_annotation': 3,
  'yellow_annotation': 3,
  'diagram_blue_purple': 6,
}

def detect_colour_annotations_by_range(im, colour_ranges=None, dilate_iterations=1, kernel_sizes=None):
  """
  Detects each colour range separately, each dilated by its own elliptical kernel
  size.

  Args:
    im (Image.Image): RGB PIL image.
    colour_ranges (dict or None): {name: (lower_hsv, upper_hsv)}. None value uses
                                  DEFAULT_COLOUR_RANGES.
    dilate_iterations (int): growth applied to each mask.
    kernel_sizes (dict or None): {name: kernel_size}. None value uses DEFAULT_KERNEL_SIZES.

  Returns:
    dict: {range_name: uint8 mask}, same (height, width) as im.
  """
  array_im = np.array(im)
  img_hsv = cv2.cvtColor(array_im, cv2.COLOR_RGB2HSV)

  if colour_ranges is None:
    colour_ranges = DEFAULT_COLOUR_RANGES
  if kernel_sizes is None:
    kernel_sizes = DEFAULT_KERNEL_SIZES

  masks = {}
  for name, (lower, upper) in colour_ranges.items():
    match = cv2.inRange(img_hsv, np.array(lower), np.array(upper))
    if dilate_iterations > 0:
      size = kernel_sizes.get(name, 3)
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
      match = cv2.dilate(match, kernel, iterations=dilate_iterations)
    masks[name] = match

  return masks


def detect_colour_annotations(im, colour_ranges=None, dilate_iterations=1, kernel_sizes=None):
  """
  Union of all per-range masks. Used as the default colour_fn for build_final_mask.
  """
  masks = detect_colour_annotations_by_range(im, colour_ranges, dilate_iterations, kernel_sizes)
  height, width = np.array(im).shape[:2]
  combined = np.zeros((height, width), dtype=np.uint8)
  for m in masks.values():
    combined = cv2.bitwise_or(combined, m)
  return combined
