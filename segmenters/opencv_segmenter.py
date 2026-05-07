from __future__ import annotations

import numpy as np
from PIL import Image

from core.layer import MaskResult
from core.mask_utils import clean_mask, dilate_mask, feather_mask, mask_to_bbox

from .base_segmenter import BaseSegmenter, SelectionRect

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


class OpenCVSegmenter(BaseSegmenter):
    """Fast local MVP segmenter based on GrabCut with rectangle fallback."""

    name = "opencv_grabcut"

    def __init__(
        self,
        grabcut_iterations: int = 3,
        min_area: int = 64,
        feather_radius: int = 1,
        fallback_to_rect: bool = True,
    ) -> None:
        self.grabcut_iterations = max(1, int(grabcut_iterations))
        self.min_area = max(0, int(min_area))
        self.feather_radius = max(0, int(feather_radius))
        self.fallback_to_rect = fallback_to_rect

    def segment(self, image: Image.Image, selection_rect: SelectionRect) -> MaskResult:
        if image.width <= 0 or image.height <= 0:
            raise ValueError("Image has invalid dimensions")

        x, y, width, height = self._clamp_rect(
            selection_rect,
            image.width,
            image.height,
        )
        if width <= 1 or height <= 1:
            raise ValueError("Selection must be at least 2x2 pixels")

        rgb = np.array(image.convert("RGB"))
        full_mask = np.zeros((image.height, image.width), dtype=np.uint8)

        if width < 4 or height < 4:
            full_mask[y : y + height, x : x + width] = 255
            return MaskResult(mask=full_mask, source=self.name)

        if cv2 is None:
            full_mask[y : y + height, x : x + width] = 255
            full_mask = clean_mask(full_mask, min_area=self.min_area)
            bbox = mask_to_bbox(full_mask)
            full_mask = feather_mask(full_mask, radius=self.feather_radius)
            return MaskResult(
                mask=full_mask,
                bbox=bbox,
                source=f"{self.name}_rect_fallback",
            )

        try:
            grabcut_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            bgd_model = np.zeros((1, 65), dtype=np.float64)
            fgd_model = np.zeros((1, 65), dtype=np.float64)
            cv2.grabCut(
                rgb,
                grabcut_mask,
                (x, y, width, height),
                bgd_model,
                fgd_model,
                self.grabcut_iterations,
                cv2.GC_INIT_WITH_RECT,
            )
            full_mask = np.where(
                (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype(np.uint8)
            full_mask[:y, :] = 0
            full_mask[y + height :, :] = 0
            full_mask[:, :x] = 0
            full_mask[:, x + width :] = 0
        except cv2.error:
            if not self.fallback_to_rect:
                raise
            full_mask[y : y + height, x : x + width] = 255

        full_mask = clean_mask(full_mask, min_area=self.min_area)
        if not np.any(full_mask) and self.fallback_to_rect:
            full_mask[y : y + height, x : x + width] = 255

        full_mask = dilate_mask(full_mask, pixels=1)
        bbox = mask_to_bbox(full_mask)
        full_mask = feather_mask(full_mask, radius=self.feather_radius)
        return MaskResult(mask=full_mask, bbox=bbox, source=self.name)

    @staticmethod
    def _clamp_rect(
        rect: SelectionRect,
        image_width: int,
        image_height: int,
    ) -> SelectionRect:
        x, y, width, height = rect
        x1 = max(0, min(image_width, int(round(x))))
        y1 = max(0, min(image_height, int(round(y))))
        x2 = max(0, min(image_width, int(round(x + width))))
        y2 = max(0, min(image_height, int(round(y + height))))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return (x1, y1, x2 - x1, y2 - y1)

