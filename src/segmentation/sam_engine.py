"""
SAM2 Engine
-----------
Core class that:
  - Loads SAM2 model (transformers backend or native sam2)
  - Caches image embeddings for repeated prompt inference
  - Supports point, box, and mask prompt modes
  - Returns mask, score, logits
  - Manages CUDA/CPU device
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class SAM2Engine:
    """Encapsulates SAM2 model loading, image embedding caching, and inference."""

    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-large",
        backend: str = "transformers",
        device: str = "cuda",
        checkpoint_path: str | None = None,
        inference_params: dict | None = None,
    ) -> None:
        """
        Args:
            model_id: HuggingFace hub model ID.
            backend: "transformers" or "native".
            device: "cuda" or "cpu".
            checkpoint_path: Local checkpoint to override model_id.
            inference_params: Dict of inference-time kwargs.
        """
        self.model_id = model_id
        self.backend = backend
        self.device = self._resolve_device(device)
        self.checkpoint_path = checkpoint_path
        self.inference_params = inference_params or {}

        self.model = None
        self.processor = None
        self._image_embedding = None
        self._image_size = None
        self._current_inputs = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model and processor into memory."""
        if self._loaded:
            logger.info("Model already loaded, skipping.")
            return
        logger.info(
            f"Loading SAM2 model '{self.model_id}' on {self.device} "
            f"(backend={self.backend})"
        )

        if self.backend == "transformers":
            self._load_transformers_backend()
        elif self.backend == "native":
            self._load_native_backend()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        self._loaded = True
        logger.info("SAM2 model loaded successfully.")

    def unload(self) -> None:
        """Free model from memory. Clears cache."""
        self.model = None
        self.processor = None
        self._image_embedding = None
        self._image_size = None
        self._current_inputs = None
        self._loaded = False
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("SAM2 model unloaded.")

    def set_image(self, image: np.ndarray | Image.Image) -> None:
        """
        Load an image and precompute its embedding. Subsequent predict()
        calls reuse this embedding without re-running the image encoder.

        Args:
            image: PIL Image or numpy array (H, W, 3) in RGB.
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        self._image_size = image.shape[:2]

        if self.backend == "transformers":
            self._set_image_transformers(image)
        elif self.backend == "native":
            self._set_image_native(image)

    def predict(
        self,
        points: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        boxes: np.ndarray | None = None,
        mask_input: np.ndarray | None = None,
        multimask_output: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run SAM2 inference with the given prompts.

        Args:
            points: (N, 2) array of (x, y) coordinates.
            point_labels: (N,) array of 1 (foreground) / 0 (background).
            boxes: (M, 4) array of (x1, y1, x2, y2) in pixel coords.
            mask_input: (1, H, W) low-res mask for refinement.
            multimask_output: Override default multimask setting.

        Returns:
            Tuple of (masks, scores, logits):
                masks:   (num_masks, H, W) bool or float array
                scores:  (num_masks,) float confidence scores
                logits:  (num_masks, H, W) raw logits
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        if self._image_embedding is None:
            raise RuntimeError("No image set. Call set_image() first.")

        if self.backend == "transformers":
            return self._predict_transformers(
                points, point_labels, boxes, mask_input, multimask_output
            )
        else:
            return self._predict_native(
                points, point_labels, boxes, mask_input, multimask_output
            )

    # ------------------------------------------------------------------
    # Internal: Device resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(preferred: str) -> str:
        if preferred == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # ------------------------------------------------------------------
    # Internal: Transformers backend
    # ------------------------------------------------------------------

    def _load_transformers_backend(self) -> None:
        from transformers import Sam2Model, Sam2Processor

        model_path = self.checkpoint_path or self.model_id
        self.processor = Sam2Processor.from_pretrained(model_path)
        self.model = Sam2Model.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def _set_image_transformers(self, image: np.ndarray) -> None:
        inputs = self.processor(
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.get_image_embeddings(inputs["pixel_values"])

        self._image_embedding = outputs  # list[torch.Tensor] (multi-scale)
        self._current_inputs = inputs

    def _predict_transformers(
        self,
        points: np.ndarray | None,
        point_labels: np.ndarray | None,
        boxes: np.ndarray | None,
        mask_input: np.ndarray | None,
        multimask_output: bool | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        multimask = (
            multimask_output
            if multimask_output is not None
            else self.inference_params.get("multimask_output", False)
        )

        # Build processor inputs
        input_points = None
        input_labels = None
        input_boxes = None

        if points is not None and point_labels is not None:
            input_points = [points[np.newaxis, :, :]]
            input_labels = [point_labels[np.newaxis, :]]

        if boxes is not None:
            input_boxes = [boxes]

        inputs = self.processor(
            original_sizes=[self._image_size],
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)

        inputs["image_embeddings"] = self._image_embedding

        with torch.inference_mode():
            outputs = self.model(**inputs, multimask_output=multimask)

        original_sizes = [self._image_size]
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            original_sizes=original_sizes,
        )[0].numpy().copy(order="C")

        scores = outputs.iou_scores.cpu().numpy()
        logits = outputs.pred_masks.cpu().numpy()

        return masks, scores, logits

    # ------------------------------------------------------------------
    # Internal: Native sam2 backend
    # ------------------------------------------------------------------

    def _load_native_backend(self) -> None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model_path = self.checkpoint_path or self.model_id
        self.model = SAM2ImagePredictor.from_pretrained(model_path)
        if self.device == "cuda":
            self.model = self.model.to("cuda")

    def _set_image_native(self, image: np.ndarray) -> None:
        self.model.set_image(image)

    def _predict_native(
        self,
        points: np.ndarray | None,
        point_labels: np.ndarray | None,
        boxes: np.ndarray | None,
        mask_input: np.ndarray | None,
        multimask_output: bool | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        multimask = (
            multimask_output
            if multimask_output is not None
            else self.inference_params.get("multimask_output", False)
        )

        masks, scores, logits = self.model.predict(
            point_coords=points,
            point_labels=point_labels,
            box=boxes,
            mask_input=mask_input,
            multimask_output=multimask,
        )
        return masks, scores, logits


class SAM2Error(Exception):
    """Base exception for SAM2-related errors."""
    pass
