from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageOps


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def load_source_image(path: str | Path) -> Image.Image:
    """Load a UI source image with EXIF orientation normalized."""
    image_path = Path(path)
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type: {image_path.suffix}")

    with Image.open(image_path) as image:
        fixed = ImageOps.exif_transpose(image)
        if fixed.mode not in {"RGB", "RGBA"}:
            fixed = fixed.convert("RGBA" if "A" in fixed.getbands() else "RGB")
        return fixed.copy()


def sanitize_filename(name: str, fallback: str = "layer") -> str:
    """Create a portable filename stem from a user-visible layer name."""
    value = name.strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-z0-9._-]+", "", value)
    value = value.strip("._-")
    return value or fallback

