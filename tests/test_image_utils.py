from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.image_utils import (
    load_image,
    resize_with_aspect_ratio,
    save_extracted_element,
)


class ImageUtilsTests(unittest.TestCase):
    def test_load_image_normalizes_grayscale_and_rgba(self) -> None:
        gray = np.array([[0, 128], [255, 64]], dtype=np.uint8)
        rgb_from_gray = load_image(gray)

        self.assertEqual(rgb_from_gray.shape, (2, 2, 3))
        self.assertEqual(rgb_from_gray.dtype, np.uint8)
        np.testing.assert_array_equal(rgb_from_gray[:, :, 0], gray)
        np.testing.assert_array_equal(rgb_from_gray[:, :, 1], gray)

        rgba = Image.new("RGBA", (2, 1), (10, 20, 30, 128))
        rgb_from_pil = load_image(rgba)

        self.assertEqual(rgb_from_pil.shape, (1, 2, 3))
        np.testing.assert_array_equal(rgb_from_pil[0, 0], np.array([10, 20, 30]))

    def test_resize_with_aspect_ratio_preserves_small_images(self) -> None:
        image = np.zeros((10, 20, 3), dtype=np.uint8)

        self.assertIs(resize_with_aspect_ratio(image, max_long_side=20), image)

        resized = resize_with_aspect_ratio(image, max_long_side=10)
        self.assertEqual(resized.shape, (5, 10, 3))

    def test_save_extracted_element_writes_png(self) -> None:
        rgba = np.zeros((2, 3, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255

        output_dir = Path.cwd() / ".test_tmp" / "image_utils"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = save_extracted_element(rgba, output_dir / "layer.png")

        self.assertTrue(output.exists())
        with Image.open(output) as saved:
            self.assertEqual(saved.mode, "RGBA")
            self.assertEqual(saved.size, (3, 2))
