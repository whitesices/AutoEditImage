from __future__ import annotations

import unittest

import numpy as np

from src.detection.grounded_dino import DetectedObject, DetectionResult
from src.pipeline.extraction_pipeline import ExtractionPipeline


class PipelineHelperTests(unittest.TestCase):
    def test_select_best_mask_handles_transformers_batch_shape(self) -> None:
        masks = np.zeros((1, 3, 4, 4), dtype=bool)
        masks[0, 2, 1, 1] = True
        scores = np.array([[0.1, 0.2, 0.9]], dtype=np.float32)

        mask, score = ExtractionPipeline._select_best_mask(masks, scores)

        self.assertEqual(mask.shape, (4, 4))
        self.assertTrue(mask[1, 1])
        self.assertAlmostEqual(score, 0.9, places=6)

    def test_select_best_mask_handles_native_shape(self) -> None:
        masks = np.zeros((3, 4, 4), dtype=bool)
        masks[1, 2, 3] = True
        scores = np.array([0.1, 0.8, 0.2], dtype=np.float32)

        mask, score = ExtractionPipeline._select_best_mask(masks, scores)

        self.assertEqual(mask.shape, (4, 4))
        self.assertTrue(mask[2, 3])
        self.assertAlmostEqual(score, 0.8, places=6)

    def test_select_detections_uses_location_hint(self) -> None:
        detections = DetectionResult(
            image_size=(100, 200),
            objects=[
                DetectedObject(
                    label="person",
                    score=0.95,
                    bbox=np.array([10, 10, 40, 90], dtype=np.float32),
                ),
                DetectedObject(
                    label="person",
                    score=0.80,
                    bbox=np.array([150, 10, 190, 90], dtype=np.float32),
                ),
            ],
        )

        selected = ExtractionPipeline._select_detections(
            detections,
            location_hint="right",
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(float(selected[0].bbox[0]), 150.0)
