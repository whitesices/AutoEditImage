"""
Image Utilities
---------------
Functions for loading, saving, and converting images.
Internal representation: PIL Image for I/O, numpy array for model operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def load_image(source: str | Path | np.ndarray | Image.Image) -> np.ndarray:
    """
    Load an image and return as RGB numpy array (H, W, 3).

    Args:
        source: File path, numpy array, or PIL Image.

    Returns:
        RGB numpy array of shape (H, W, 3), dtype uint8.
    """
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.shape[2] == 1:
            arr = np.concatenate([arr] * 3, axis=-1)
        return arr.astype(np.uint8)

    if isinstance(source, Image.Image):
        return np.array(source.convert("RGB"))

    # str or Path
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    pil_img = Image.open(str(path)).convert("RGB")
    return np.array(pil_img)


def save_extracted_element(
    rgba_image: np.ndarray,
    output_path: str | Path,
) -> Path:
    """
    Save an RGBA extracted element as a PNG with transparency.

    Args:
        rgba_image: (H, W, 4) RGBA numpy array.
        output_path: Destination path.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pil_img = Image.fromarray(rgba_image, mode="RGBA")
    pil_img.save(str(output_path), format="PNG")

    return output_path


def resize_with_aspect_ratio(
    image: np.ndarray,
    max_long_side: int,
) -> np.ndarray:
    """
    Resize image so the longest side is at most max_long_side,
    preserving aspect ratio. Returns original if already small enough.
    """
    h, w = image.shape[:2]
    current_long = max(h, w)
    if current_long <= max_long_side:
        return image

    scale = max_long_side / current_long
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
