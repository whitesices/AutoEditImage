"""
SAM Prompts
-----------
Clean classes and builders for constructing SAM2 prompts:
  - PointPrompt:  (x, y, label) for positive/negative clicks
  - BoxPrompt:    (x1, y1, x2, y2) for bounding box prompts
  - MaskPrompt:   coarse mask for refinement
  - PromptSet:    combines multiple prompt types for a single prediction call
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np


# ======================================================================
# Individual Prompt Types
# ======================================================================

@dataclass(frozen=True)
class PointPrompt:
    """A single point prompt. label=1 for foreground, 0 for background."""

    x: float
    y: float
    label: int = 1

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    def validate(self) -> None:
        if self.label not in (0, 1):
            raise ValueError(f"Point label must be 0 or 1, got {self.label}")


@dataclass(frozen=True)
class BoxPrompt:
    """A bounding box prompt.(x1,y1) top-left, (x2,y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def validate(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(
                f"Invalid box: ({self.x1}, {self.y1}) -> ({self.x2}, {self.y2})"
            )


@dataclass(frozen=True)
class MaskPrompt:
    """A coarse mask prompt for refinement."""

    mask: np.ndarray

    def validate(self) -> None:
        if self.mask.ndim not in (2, 3):
            raise ValueError(f"Mask must be 2D or 3D, got shape {self.mask.shape}")


# ======================================================================
# PromptSet (Composite)
# ======================================================================

class PromptSet:
    """
    A collection of prompts for a single object or joint prediction.

    Supports:
      - Multiple point prompts
      - One or more box prompts
      - One mask prompt (for refinement)
    """

    def __init__(self) -> None:
        self._points: list[PointPrompt] = []
        self._boxes: list[BoxPrompt] = []
        self._mask: MaskPrompt | None = None

    # ---- Builders (fluent chaining) ----

    def add_point(self, x: float, y: float, label: int = 1) -> Self:
        p = PointPrompt(x, y, label)
        p.validate()
        self._points.append(p)
        return self

    def add_box(self, x1: float, y1: float, x2: float, y2: float) -> Self:
        b = BoxPrompt(x1, y1, x2, y2)
        b.validate()
        self._boxes.append(b)
        return self

    def set_mask(self, mask: np.ndarray) -> Self:
        m = MaskPrompt(mask)
        m.validate()
        self._mask = m
        return self

    # ---- Output (what SAM2Engine.predict() expects) ----

    def get_points(self) -> np.ndarray | None:
        """Return (N, 2) array or None."""
        if not self._points:
            return None
        return np.array([p.as_array() for p in self._points], dtype=np.float32)

    def get_point_labels(self) -> np.ndarray | None:
        """Return (N,) array or None."""
        if not self._points:
            return None
        return np.array([p.label for p in self._points], dtype=np.int32)

    def get_boxes(self) -> np.ndarray | None:
        """Return (M, 4) array or None."""
        if not self._boxes:
            return None
        return np.array([b.as_array() for b in self._boxes], dtype=np.float32)

    def get_mask(self) -> np.ndarray | None:
        """Return (H, W) mask or None."""
        return self._mask.mask if self._mask else None

    def has_any_prompt(self) -> bool:
        return bool(self._points or self._boxes or self._mask)

    def clear(self) -> None:
        self._points.clear()
        self._boxes.clear()
        self._mask = None

    def __len__(self) -> int:
        return len(self._points) + len(self._boxes) + (1 if self._mask else 0)

    def __repr__(self) -> str:
        return (
            f"PromptSet(points={len(self._points)}, "
            f"boxes={len(self._boxes)}, "
            f"mask={'yes' if self._mask else 'no'})"
        )
