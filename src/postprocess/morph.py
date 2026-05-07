"""
Mask Post-processing
--------------------
Morphological operations to clean up SAM2 output masks:
  - Binary thresholding
  - Close operation (fill small holes)
  - Open operation (remove noise)
  - Connected component filtering (remove small fragments)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def refine_mask(
    mask: np.ndarray,
    threshold: float = 0.5,
    close_kernel: int = 5,
    open_kernel: int = 3,
    min_contour_ratio: float = 0.01,
) -> np.ndarray:
    """
    Apply post-processing to a raw SAM2 mask.

    Pipeline:
      1. Threshold to binary
      2. Morphological close (fill holes)
      3. Morphological open (remove noise)
      4. Connected component filter (remove tiny islands)

    Args:
        mask: Input mask (H, W) float or bool.
        threshold: Threshold for binarization if mask is float.
        close_kernel: Kernel size for closing. 0 to skip.
        open_kernel: Kernel size for opening. 0 to skip.
        min_contour_ratio: Minimum contour area ratio, 0 to skip.

    Returns:
        Refined binary mask (H, W) as bool.
    """
    # 1. Binarize
    if mask.dtype != bool:
        binary = mask > threshold
    else:
        binary = mask.copy()

    binary_u8 = binary.astype(np.uint8) * 255

    # 2. Morphological close (fill holes)
    if close_kernel and close_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
        )
        binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, kernel)

    # 3. Morphological open (remove noise)
    if open_kernel and open_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)
        )
        binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, kernel)

    # 4. Connected component filtering
    if min_contour_ratio > 0.0:
        binary_u8 = _filter_small_components(binary_u8, min_contour_ratio)

    return binary_u8.astype(bool)


def _filter_small_components(
    binary: np.ndarray,
    min_area_ratio: float,
) -> np.ndarray:
    """Remove connected components with area below min_area_ratio * max_area."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if num_labels <= 1:
        return binary

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = areas.max()

    if max_area == 0:
        return np.zeros_like(binary)

    min_area = max_area * min_area_ratio

    result = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result[labels == i] = 255

    return result


def mask_to_trimap(
    mask: np.ndarray,
    erosion_kernel: int = 5,
    dilation_kernel: int = 5,
) -> np.ndarray:
    """
    Convert a binary mask to a trimap for alpha matting.

    Values:
      0   = definite background
      128 = uncertain (edge region)
      255 = definite foreground

    This is a placeholder for Phase 2 matting integration.
    """
    binary_u8 = mask.astype(np.uint8) * 255

    kernel_e = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erosion_kernel, erosion_kernel)
    )
    kernel_d = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_kernel, dilation_kernel)
    )

    eroded = cv2.erode(binary_u8, kernel_e, iterations=1)
    dilated = cv2.dilate(binary_u8, kernel_d, iterations=1)

    trimap = np.full_like(binary_u8, 128)
    trimap[eroded > 0] = 255
    trimap[dilated == 0] = 0

    return trimap
