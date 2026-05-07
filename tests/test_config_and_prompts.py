from __future__ import annotations

import unittest

import numpy as np

from src import load_config
from src.segmentation.sam_prompts import PromptSet


class ConfigAndPromptTests(unittest.TestCase):
    def test_default_config_has_core_sections(self) -> None:
        config = load_config()

        self.assertIn("models", config)
        self.assertIn("sam2", config["models"])
        self.assertIn("grounding_dino", config["models"])
        self.assertTrue(config["output"]["default_dir"])

    def test_prompt_set_builds_arrays_and_validates_labels(self) -> None:
        prompts = PromptSet().add_point(10, 20, 1).add_box(1, 2, 30, 40)

        np.testing.assert_array_equal(
            prompts.get_points(),
            np.array([[10, 20]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            prompts.get_point_labels(),
            np.array([1], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            prompts.get_boxes(),
            np.array([[1, 2, 30, 40]], dtype=np.float32),
        )
        self.assertTrue(prompts.has_any_prompt())
        self.assertEqual(len(prompts), 2)

        with self.assertRaisesRegex(ValueError, "Point label"):
            PromptSet().add_point(0, 0, 2)

        with self.assertRaisesRegex(ValueError, "Invalid box"):
            PromptSet().add_box(10, 10, 5, 5)
