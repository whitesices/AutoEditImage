"""
Extraction Pipeline
-------------------
Orchestrates the full extraction workflow:
  1. Load image
  2. Detect objects via Grounding DINO from text query
  3. Convert detections to SAM2 prompts
  4. Segment via SAM2
  5. Post-process masks
  6. Return structured results and save outputs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.detection.grounded_dino import GroundingDINO, DetectionResult, DetectedObject
from src.segmentation.sam_engine import SAM2Engine
from src.segmentation.sam_prompts import PromptSet
from src.postprocess.morph import refine_mask
from src.utils.image_utils import load_image, save_extracted_element
from src.utils.visualization import draw_mask_overlay, draw_detections

logger = logging.getLogger(__name__)


@dataclass
class ExtractedElement:
    """A single extracted element from the pipeline."""

    mask: np.ndarray
    score: float
    bbox: np.ndarray
    label: str
    extracted_image: np.ndarray | None = None
    original_image: np.ndarray | None = None


@dataclass
class PipelineResult:
    """Container for pipeline outputs."""

    elements: list[ExtractedElement] = field(default_factory=list)
    image_size: tuple[int, int] | None = None
    source_path: str | None = None
    detections: DetectionResult | None = None

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, idx: int) -> ExtractedElement:
        return self.elements[idx]


class ExtractionPipeline:
    """
    Top-level orchestrator for intelligent element extraction.

    Usage:
        pipeline = ExtractionPipeline(config)
        pipeline.load_models()
        result = pipeline.extract_text("image.jpg", "a red car")
        pipeline.save_results(result)
        pipeline.unload_models()
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.sam2: SAM2Engine | None = None
        self.dino: GroundingDINO | None = None
        self._models_loaded = False

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Initialize and load all required models."""
        sam2_cfg = self.config.get("models", {}).get("sam2", {})
        dino_cfg = self.config.get("models", {}).get("grounding_dino", {})
        device_cfg = self.config.get("device", {})
        device = device_cfg.get("preferred", "cuda")

        logger.info("Loading extraction pipeline models...")

        self.sam2 = SAM2Engine(
            model_id=sam2_cfg.get("model_id", "facebook/sam2-hiera-large"),
            backend=sam2_cfg.get("backend", "transformers"),
            device=device,
            checkpoint_path=sam2_cfg.get("checkpoint_path"),
            inference_params=sam2_cfg.get("inference", {}),
        )
        self.sam2.load()

        self.dino = GroundingDINO(
            model_id=dino_cfg.get("model_id", "IDEA-Research/grounding-dino-base"),
            device=device,
            checkpoint_path=dino_cfg.get("checkpoint_path"),
            box_threshold=dino_cfg.get("box_threshold", 0.35),
            text_threshold=dino_cfg.get("text_threshold", 0.25),
        )
        self.dino.load()

        self._models_loaded = True
        logger.info("All models loaded.")

    def unload_models(self) -> None:
        """Free all models from memory."""
        if self.sam2:
            self.sam2.unload()
        if self.dino:
            self.dino.unload()
        self._models_loaded = False
        logger.info("All models unloaded.")

    # ------------------------------------------------------------------
    # Extraction modes
    # ------------------------------------------------------------------

    def extract_text(
        self,
        image_path: str | Path | np.ndarray | Image.Image,
        text_query: str,
        apply_postprocess: bool = True,
        auto_lowercase: bool = True,
        location_hint: str | None = None,
        max_results: int | None = None,
    ) -> PipelineResult:
        """
        Text-driven extraction: detect objects via text, then segment via SAM2.

        Args:
            image_path: Path to image, numpy array, or PIL Image.
            text_query: Natural language query (e.g., "a red car", "a person").
            apply_postprocess: Whether to apply morphological refinement to masks.
            auto_lowercase: Auto-format the text query for Grounding DINO.

        Returns:
            PipelineResult with extracted elements.
        """
        # 1. Load image
        image = load_image(image_path)
        h, w = image.shape[:2]

        # 2. Detect objects from text
        query = text_query.lower().strip() if auto_lowercase else text_query
        detections = self.dino.detect(image, query)

        if len(detections) == 0:
            logger.warning(f"No objects detected for query: '{text_query}'")
            return PipelineResult(
                elements=[],
                image_size=(h, w),
                source_path=str(image_path)
                if isinstance(image_path, (str, Path))
                else None,
                detections=detections,
            )

        selected_detections = self._select_detections(
            detections,
            location_hint=location_hint,
            max_results=max_results,
        )

        # 3. Precompute image embedding for SAM2 (shared across all objects)
        self.sam2.set_image(image)

        # 4. For each detected object, segment via SAM2
        elements: list[ExtractedElement] = []
        multimask = (
            self.config.get("models", {})
            .get("sam2", {})
            .get("inference", {})
            .get("multimask_output", False)
        )

        for obj in selected_detections.objects:
            prompts = PromptSet()
            prompts.add_box(
                x1=float(obj.bbox[0]),
                y1=float(obj.bbox[1]),
                x2=float(obj.bbox[2]),
                y2=float(obj.bbox[3]),
            )

            masks, scores, _logits = self.sam2.predict(
                boxes=prompts.get_boxes(),
                multimask_output=multimask,
            )

            mask, score = self._select_best_mask(masks, scores)

            # 5. Post-process mask
            if apply_postprocess:
                morph_cfg = self.config.get("postprocess", {}).get("morph", {})
                mask = refine_mask(
                    mask,
                    close_kernel=morph_cfg.get("close_kernel_size", 5),
                    open_kernel=morph_cfg.get("open_kernel_size", 3),
                    min_contour_ratio=morph_cfg.get("min_contour_area_ratio", 0.01),
                )

            element = ExtractedElement(
                mask=mask,
                score=score,
                bbox=obj.bbox,
                label=obj.label,
                original_image=image,
            )
            element.extracted_image = self._apply_mask_to_image(image, mask)
            elements.append(element)

        return PipelineResult(
            elements=elements,
            image_size=(h, w),
            source_path=str(image_path)
            if isinstance(image_path, (str, Path))
            else None,
            detections=detections,
        )

    def extract_point(
        self,
        image_path: str | Path | np.ndarray | Image.Image,
        points: list[tuple[float, float]],
        labels: list[int] | None = None,
        apply_postprocess: bool = True,
    ) -> PipelineResult:
        """
        Point-driven extraction: click to segment.

        Args:
            image_path: Path or image array.
            points: List of (x, y) click coordinates.
            labels: List of 1 (foreground) / 0 (background). Default: all 1.
            apply_postprocess: Whether to apply morphological refinement.
        """
        image = load_image(image_path)
        h, w = image.shape[:2]

        if labels is None:
            labels = [1] * len(points)

        self.sam2.set_image(image)

        prompts = PromptSet()
        for (x, y), lbl in zip(points, labels):
            prompts.add_point(x, y, lbl)

        masks, scores, _logits = self.sam2.predict(
            points=prompts.get_points(),
            point_labels=prompts.get_point_labels(),
        )

        mask, score = self._select_best_mask(masks, scores)

        if apply_postprocess:
            morph_cfg = self.config.get("postprocess", {}).get("morph", {})
            mask = refine_mask(
                mask,
                close_kernel=morph_cfg.get("close_kernel_size", 5),
                open_kernel=morph_cfg.get("open_kernel_size", 3),
                min_contour_ratio=morph_cfg.get("min_contour_area_ratio", 0.01),
            )

        # Compute bbox from mask
        ys, xs = np.where(mask)
        if len(xs) > 0 and len(ys) > 0:
            bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
        else:
            bbox = np.array([0, 0, 0, 0], dtype=np.float32)

        element = ExtractedElement(
            mask=mask,
            score=score,
            bbox=bbox,
            label="point_selection",
            original_image=image,
            extracted_image=self._apply_mask_to_image(image, mask),
        )

        return PipelineResult(
            elements=[element],
            image_size=(h, w),
            source_path=str(image_path)
            if isinstance(image_path, (str, Path))
            else None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_detections(
        detections: DetectionResult,
        location_hint: str | None = None,
        max_results: int | None = None,
    ) -> DetectionResult:
        """Filter/sort detections by source-canvas location hints."""
        objects = list(detections.objects)
        if not objects:
            return DetectionResult(image_size=detections.image_size, objects=[])

        normalized_hint = (location_hint or "").strip().lower().replace("-", "_")
        if normalized_hint:
            objects.sort(
                key=lambda obj: ExtractionPipeline._location_score(
                    obj.bbox,
                    detections.image_size,
                    normalized_hint,
                )
            )
            if max_results is None:
                max_results = 1
        else:
            objects.sort(key=lambda obj: obj.score, reverse=True)

        if max_results is not None and max_results > 0:
            objects = objects[:max_results]

        return DetectionResult(image_size=detections.image_size, objects=objects)

    @staticmethod
    def _location_score(
        bbox: np.ndarray,
        image_size: tuple[int, int],
        location_hint: str,
    ) -> float:
        """Lower score means closer to the requested region."""
        height, width = image_size
        x1, y1, x2, y2 = [float(value) for value in bbox]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        norm_x = cx / max(1.0, float(width))
        norm_y = cy / max(1.0, float(height))

        if location_hint == "left":
            return norm_x
        if location_hint == "right":
            return 1.0 - norm_x
        if location_hint == "top":
            return norm_y
        if location_hint == "bottom":
            return 1.0 - norm_y
        if location_hint == "center":
            return (norm_x - 0.5) ** 2 + (norm_y - 0.5) ** 2

        anchors = {
            "top_left": (0.0, 0.0),
            "top_right": (1.0, 0.0),
            "bottom_left": (0.0, 1.0),
            "bottom_right": (1.0, 1.0),
        }
        if location_hint in anchors:
            anchor_x, anchor_y = anchors[location_hint]
            return (norm_x - anchor_x) ** 2 + (norm_y - anchor_y) ** 2

        return 0.0

    @staticmethod
    def _select_best_mask(
        masks: np.ndarray,
        scores: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Select the highest-scoring 2D mask from SAM2 outputs.

        Transformers 5.x can return masks shaped (B, N, H, W), while the
        native backend commonly returns (N, H, W). Scores follow the same
        leading dimensions. The rest of the pipeline expects a squeezed 2D
        mask before OpenCV post-processing.
        """
        masks_arr = np.asarray(masks)
        scores_arr = np.asarray(scores)

        if scores_arr.size == 0:
            raise ValueError("Cannot select a mask from empty scores.")

        best_flat_idx = int(np.argmax(scores_arr))

        if masks_arr.ndim == 4:
            batch_count, mask_count = masks_arr.shape[:2]
            if scores_arr.ndim >= 2 and scores_arr.shape[:2] == (
                batch_count,
                mask_count,
            ):
                batch_idx, mask_idx = np.unravel_index(
                    best_flat_idx,
                    scores_arr.shape[:2],
                )
            elif batch_count == 1:
                batch_idx = 0
                mask_idx = best_flat_idx % mask_count
            else:
                batch_idx, mask_idx = divmod(best_flat_idx, mask_count)
            mask = masks_arr[batch_idx, mask_idx]
        elif masks_arr.ndim == 3:
            mask_idx = best_flat_idx % masks_arr.shape[0]
            mask = masks_arr[mask_idx]
        elif masks_arr.ndim == 2:
            mask = masks_arr
        else:
            raise ValueError(f"Unsupported mask shape: {masks_arr.shape}")

        return np.squeeze(mask), float(scores_arr.flat[best_flat_idx])

    @staticmethod
    def _apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply a boolean mask to an RGB image to produce RGBA with
        transparent background."""
        if mask.dtype != bool:
            mask = mask > 0.5
        rgba = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
        rgba[:, :, :3] = image
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)
        return rgba

    def save_results(
        self,
        result: PipelineResult,
        output_dir: str | Path = "./outputs",
        save_overlay: bool | None = None,
    ) -> list[Path]:
        """
        Save pipeline results to disk.

        Returns:
            List of saved file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: list[Path] = []
        cfg = self.config.get("output", {})

        for i, element in enumerate(result.elements):
            # Save individual element as RGBA PNG
            safe_label = element.label.replace(" ", "_")
            element_path = output_dir / f"element_{i:04d}_{safe_label}.png"
            save_extracted_element(element.extracted_image, element_path)
            saved_paths.append(element_path)

            # Save overlay visualization
            do_overlay = save_overlay if save_overlay is not None else cfg.get(
                "save_overlay", True
            )
            if do_overlay:
                overlay = draw_mask_overlay(
                    element.original_image,
                    element.mask,
                    alpha=cfg.get("overlay_alpha", 0.5),
                )
                overlay_path = (
                    output_dir / f"overlay_{i:04d}_{safe_label}.png"
                )
                Image.fromarray(overlay).save(str(overlay_path))
                saved_paths.append(overlay_path)

        return saved_paths
