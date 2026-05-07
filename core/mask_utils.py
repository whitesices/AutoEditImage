from __future__ import annotations

from typing import TypeAlias

import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


BBox: TypeAlias = tuple[int, int, int, int]


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    """Return a 2D uint8 alpha mask in the range 0..255."""
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            max_value = float(np.nanmax(array)) if array.size else 0.0
            if max_value <= 1.0:
                array = array * 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def mask_to_bbox(mask: np.ndarray) -> BBox | None:
    """Compute the tight `(x, y, width, height)` bbox of non-zero pixels."""
    normalized = normalize_mask(mask)
    ys, xs = np.nonzero(normalized > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def clean_mask(mask: np.ndarray, min_area: int = 64) -> np.ndarray:
    """Remove tiny connected components and lightly close small holes."""
    normalized = normalize_mask(mask)
    binary = np.where(normalized > 0, 255, 0).astype(np.uint8)
    if not np.any(binary):
        return binary

    threshold = max(0, int(min_area))
    if cv2 is not None:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        cleaned = np.zeros_like(binary)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= threshold:
                cleaned[labels == label] = 255
    else:
        cleaned = _connected_components_filter(binary, threshold)

    if not np.any(cleaned):
        return cleaned

    if cv2 is not None:
        kernel = np.ones((3, 3), dtype=np.uint8)
        return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return erode_mask(dilate_mask(cleaned, pixels=1), pixels=1)


def feather_mask(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Soften mask edges with a Gaussian blur."""
    normalized = normalize_mask(mask)
    radius = max(0, int(radius))
    if radius == 0 or not np.any(normalized):
        return normalized
    kernel_size = radius * 2 + 1
    if cv2 is not None:
        return cv2.GaussianBlur(
            normalized,
            (kernel_size, kernel_size),
            sigmaX=radius / 2,
        )
    return np.array(
        Image.fromarray(normalized, mode="L").filter(ImageFilter.GaussianBlur(radius))
    )


def dilate_mask(mask: np.ndarray, pixels: int = 1) -> np.ndarray:
    """Grow mask boundaries by a number of pixels."""
    normalized = normalize_mask(mask)
    pixels = max(0, int(pixels))
    if pixels == 0 or not np.any(normalized):
        return normalized
    kernel_size = pixels * 2 + 1
    if cv2 is not None:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.dilate(normalized, kernel, iterations=1)
    return np.array(
        Image.fromarray(normalized, mode="L").filter(ImageFilter.MaxFilter(kernel_size))
    )


def erode_mask(mask: np.ndarray, pixels: int = 1) -> np.ndarray:
    """Shrink mask boundaries by a number of pixels."""
    normalized = normalize_mask(mask)
    pixels = max(0, int(pixels))
    if pixels == 0 or not np.any(normalized):
        return normalized
    kernel_size = pixels * 2 + 1
    if cv2 is not None:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.erode(normalized, kernel, iterations=1)
    return np.array(
        Image.fromarray(normalized, mode="L").filter(ImageFilter.MinFilter(kernel_size))
    )


def apply_mask_to_rgba(image: Image.Image, mask: np.ndarray, bbox: BBox) -> Image.Image:
    """Crop an image to bbox and apply the cropped mask as alpha."""
    normalized = normalize_mask(mask)
    if normalized.shape[:2] != (image.height, image.width):
        raise ValueError(
            f"Mask shape {normalized.shape[:2]} does not match image size "
            f"{image.width}x{image.height}"
        )

    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        raise ValueError("bbox width and height must be positive")
    if x < 0 or y < 0 or x + width > image.width or y + height > image.height:
        raise ValueError("bbox is outside the image bounds")

    rgba = image.convert("RGBA")
    image_crop = rgba.crop((x, y, x + width, y + height))
    alpha_crop = Image.fromarray(normalized[y : y + height, x : x + width], mode="L")
    image_crop.putalpha(alpha_crop)
    return image_crop


def _connected_components_filter(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Small numpy fallback for environments where OpenCV is not installed."""
    height, width = binary.shape
    foreground = binary > 0
    visited = np.zeros_like(foreground, dtype=bool)
    output = np.zeros_like(binary)
    neighbors = (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )

    for y in range(height):
        for x in range(width):
            if not foreground[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                px, py = stack.pop()
                pixels.append((px, py))
                for dx, dy in neighbors:
                    nx = px + dx
                    ny = py + dy
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    if foreground[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            if len(pixels) >= min_area:
                for px, py in pixels:
                    output[py, px] = 255
    return output

