from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from core.exporter import LayerExporter
from core.layer import MaskResult
from core.mask_utils import apply_mask_to_rgba, clean_mask, mask_to_bbox, normalize_mask
from core.project import ProjectData
from segmenters.opencv_segmenter import OpenCVSegmenter


class LayerCoreTests(unittest.TestCase):
    def test_mask_bbox_and_rgba_crop_contract(self) -> None:
        image = Image.new("RGB", (8, 6), (10, 20, 30))
        mask = np.zeros((6, 8), dtype=np.float32)
        mask[2:5, 3:7] = 1.0

        normalized = normalize_mask(mask)
        bbox = mask_to_bbox(normalized)
        self.assertEqual(bbox, (3, 2, 4, 3))

        rgba = apply_mask_to_rgba(image, normalized, bbox)
        self.assertEqual(rgba.mode, "RGBA")
        self.assertEqual(rgba.size, (4, 3))
        self.assertEqual(rgba.getchannel("A").getbbox(), (0, 0, 4, 3))

    def test_clean_mask_removes_tiny_islands(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:10, 2:10] = 255
        mask[18, 18] = 255

        cleaned = clean_mask(mask, min_area=10)

        self.assertEqual(cleaned.dtype, np.uint8)
        self.assertEqual(cleaned[4, 4], 255)
        self.assertEqual(cleaned[18, 18], 0)

    def test_project_export_contract(self) -> None:
        project = ProjectData()
        image = Image.new("RGB", (10, 8), (100, 120, 140))
        project.set_source_image("source.png", image)

        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[1:5, 2:7] = 255
        layer = project.add_layer_from_mask("Hero Layer", MaskResult(mask=mask))

        output_dir = Path.cwd() / ".test_tmp" / "layer_export"
        result = LayerExporter(output_dir).export_all_layers(project)

        self.assertTrue((output_dir / "layers" / "001_hero_layer.png").exists())
        self.assertTrue((output_dir / "masks" / "001_hero_layer_mask.png").exists())
        self.assertTrue((output_dir / "preview.png").exists())
        self.assertTrue((output_dir / "project.json").exists())
        self.assertEqual(layer.file, "layers/001_hero_layer.png")
        self.assertEqual(layer.mask_file, "masks/001_hero_layer_mask.png")
        self.assertIn("project", result)

        data = json.loads((output_dir / "project.json").read_text(encoding="utf-8"))
        self.assertEqual(data["canvas"], {"width": 10, "height": 8})
        self.assertEqual(data["layers"][0]["x"], 2)
        self.assertEqual(data["layers"][0]["width"], 5)

    def test_opencv_segmenter_returns_full_canvas_mask(self) -> None:
        image = Image.new("RGB", (32, 24), (30, 30, 30))
        segmenter = OpenCVSegmenter(grabcut_iterations=1, min_area=1)

        result = segmenter.segment(image, (5, 4, 12, 10))

        self.assertEqual(result.mask.shape, (24, 32))
        self.assertFalse(result.is_empty)
        self.assertIsNotNone(result.bbox)

