"""
Visualization Utilities
-----------------------
Functions for drawing mask overlays, bounding boxes, and debug visualizations.
"""

from __future__ import annotations

import numpy as np
import cv2


def draw_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Draw a semi-transparent mask overlay on the image.

    Args:
        image: (H, W, 3) RGB array.
        mask: (H, W) boolean or float mask.
        color: RGB color for the overlay.
        alpha: Transparency of the overlay (0=transparent, 1=opaque).

    Returns:
        (H, W, 3) RGB array with overlay.
    """
    overlay = image.copy().astype(np.float32)
    mask_bool = mask > 0.5 if mask.dtype != bool else mask

    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
            overlay[:, :, c],
        )

    return overlay.astype(np.uint8)


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: list[str] | None = None,
    scores: list[float] | None = None,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    Draw bounding boxes (and optionally labels + scores) on an image.

    Args:
        image: (H, W, 3) RGB array.
        boxes: (N, 4) array of [x1, y1, x2, y2].
        labels: Optional list of N text labels.
        scores: Optional list of N confidence scores.
        color: RGB color for boxes.

    Returns:
        (H, W, 3) RGB array with drawn boxes.
    """
    bgr_color = (color[2], color[1], color[0])
    vis = image.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), bgr_color, 2)

        if labels or scores:
            text_parts = []
            if labels and i < len(labels):
                text_parts.append(labels[i])
            if scores and i < len(scores):
                text_parts.append(f"{scores[i]:.2f}")
            text = ": ".join(text_parts)

            cv2.putText(
                vis, text, (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 1,
            )

    return vis


def create_debug_grid(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: np.ndarray | None = None,
) -> np.ndarray:
    """
    Create a 2-panel debug view: original | overlay.

    Returns:
        (H, 2*W, 3) side-by-side numpy array.
    """
    overlay = draw_mask_overlay(image, mask)
    grid = np.concatenate([image, overlay], axis=1)
    return grid
