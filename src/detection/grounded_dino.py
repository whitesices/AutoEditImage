"""
Grounding DINO Integration
--------------------------
Wraps IDEA-Research/grounding-dino-base via HuggingFace transformers.
Takes a text query + image, returns detected bounding boxes with labels and scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    """A single detected object with bounding box and metadata."""

    label: str
    score: float
    bbox: np.ndarray  # (4,) [x1, y1, x2, y2] in pixel coords

    @property
    def xyxy(self) -> np.ndarray:
        return self.bbox

    @property
    def xywh(self) -> np.ndarray:
        """Convert to (x_center, y_center, width, height)."""
        x1, y1, x2, y2 = self.bbox
        return np.array([
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            x2 - x1,
            y2 - y1,
        ])


@dataclass
class DetectionResult:
    """Container for all detections from a single image + text query."""

    image_size: tuple[int, int]  # (H, W)
    objects: list[DetectedObject] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.objects)

    def __getitem__(self, idx: int) -> DetectedObject:
        return self.objects[idx]

    def top_k(self, k: int) -> DetectionResult:
        """Return the top-k highest scoring objects."""
        sorted_objs = sorted(self.objects, key=lambda o: o.score, reverse=True)
        return DetectionResult(
            image_size=self.image_size,
            objects=sorted_objs[:k],
        )


class GroundingDINO:
    """
    Zero-shot open-vocabulary object detector.
    Uses AutoModelForZeroShotObjectDetection from HuggingFace transformers.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda",
        checkpoint_path: str | None = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> None:
        self.model_id = model_id
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.checkpoint_path = checkpoint_path
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> None:
        """Load model and processor from HuggingFace."""
        if self._loaded:
            return
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        model_path = self.checkpoint_path or self.model_id
        logger.info(f"Loading Grounding DINO from '{model_path}' on {self.device}")

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path
        ).to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info("Grounding DINO loaded successfully.")

    def unload(self) -> None:
        """Free model from memory."""
        self.model = None
        self.processor = None
        self._loaded = False
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("Grounding DINO unloaded.")

    def detect(
        self,
        image: np.ndarray | Image.Image,
        text: str,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> DetectionResult:
        """
        Detect objects matching the text query.

        Args:
            image: PIL Image or numpy array (H, W, 3) in RGB.
            text: Text query. Multiple phrases separated by ". ".
                  Example: "a cat. a dog. a person."
                  The trailing period is required by Grounding DINO.
            box_threshold: Override default confidence threshold.
            text_threshold: Override default text alignment threshold.

        Returns:
            DetectionResult with zero or more DetectedObject instances.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Ensure text ends with a period
        text = text.strip()
        if not text.endswith("."):
            text += "."

        box_thr = box_threshold if box_threshold is not None else self.box_threshold
        text_thr = text_threshold if text_threshold is not None else self.text_threshold

        inputs = self.processor(
            images=image, text=text, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process: target_sizes expects (height, width)
        h, w = image.size[1], image.size[0]
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_thr,
            text_threshold=text_thr,
            target_sizes=[(h, w)],
        )[0]

        objects = []
        for label, score, box in zip(
            results["text_labels"], results["scores"], results["boxes"]
        ):
            bbox = box.cpu().numpy()
            objects.append(
                DetectedObject(
                    label=str(label),
                    score=float(score),
                    bbox=bbox,
                )
            )

        return DetectionResult(image_size=(h, w), objects=objects)

    def detect_auto_text(
        self, image: np.ndarray | Image.Image, text: str
    ) -> DetectionResult:
        """
        Convenience wrapper that auto-lowercases and formats the query.
        For multiple objects, use detect() with proper ". " formatting.
        """
        formatted = text.lower().strip()
        return self.detect(image, formatted)
