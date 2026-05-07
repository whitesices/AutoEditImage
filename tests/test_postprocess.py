from __future__ import annotations

import unittest

import numpy as np

from src.postprocess.morph import mask_to_trimap, refine_mask


class PostprocessTests(unittest.TestCase):
    def test_refine_mask_removes_small_components_relative_to_largest(self) -> None:
        mask = np.zeros((24, 24), dtype=np.float32)
        mask[2:14, 2:14] = 1.0
        mask[21, 21] = 1.0

        refined = refine_mask(
            mask,
            close_kernel=0,
            open_kernel=0,
            min_contour_ratio=0.2,
        )

        self.assertEqual(refined.dtype, bool)
        self.assertTrue(refined[5, 5])
        self.assertFalse(refined[21, 21])

    def test_mask_to_trimap_uses_expected_alpha_values(self) -> None:
        mask = np.zeros((9, 9), dtype=bool)
        mask[3:6, 3:6] = True

        trimap = mask_to_trimap(mask, erosion_kernel=3, dilation_kernel=3)

        self.assertEqual(trimap.shape, mask.shape)
        self.assertTrue(set(np.unique(trimap)).issubset({0, 128, 255}))
        self.assertEqual(trimap[4, 4], 255)
        self.assertEqual(trimap[0, 0], 0)
