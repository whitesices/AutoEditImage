from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .mask_utils import BBox, mask_to_bbox, normalize_mask


@dataclass(slots=True)
class MaskResult:
    """Segmentation output aligned to the source canvas."""

    mask: np.ndarray
    bbox: BBox | None = None
    score: float | None = None
    source: str = "manual"

    def __post_init__(self) -> None:
        self.mask = normalize_mask(self.mask)
        if self.mask.ndim != 2:
            raise ValueError("MaskResult.mask must be a 2D array")
        if self.bbox is None:
            self.bbox = mask_to_bbox(self.mask)

    @property
    def is_empty(self) -> bool:
        return self.bbox is None or not np.any(self.mask)


@dataclass(slots=True)
class LayerItem:
    """One extracted layer and its source-canvas metadata."""

    id: str
    name: str
    mask: np.ndarray
    bbox: BBox
    visible: bool = True
    opacity: float = 1.0
    blend_mode: str = "normal"
    file: str = ""
    mask_file: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mask = normalize_mask(self.mask)
        if self.mask.ndim != 2:
            raise ValueError("LayerItem.mask must be a 2D array")
        x, y, width, height = self.bbox
        if width <= 0 or height <= 0:
            raise ValueError("LayerItem.bbox width and height must be positive")
        if x < 0 or y < 0:
            raise ValueError("LayerItem.bbox x and y must be non-negative")
        self.opacity = max(0.0, min(1.0, float(self.opacity)))

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]

    def rename(self, name: str) -> None:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Layer name cannot be empty")
        self.name = clean_name

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)

    def to_project_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "visible": self.visible,
            "file": self.file,
            "mask": self.mask_file,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "opacity": self.opacity,
            "blend_mode": self.blend_mode,
        }

