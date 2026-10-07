from __future__ import annotations

import unittest

import numpy as np

from plant_counter.preprocess import enhance_green


class GreenEnhancementTests(unittest.TestCase):
    def test_enhances_only_green_dominant_pixels(self) -> None:
        image = np.array([[[40, 120, 50], [80, 80, 80], [150, 60, 40]]], dtype=np.uint8)

        enhanced = enhance_green(image)

        self.assertGreater(enhanced[0, 0, 1], image[0, 0, 1])
        np.testing.assert_array_equal(enhanced[0, 1], image[0, 1])
        np.testing.assert_array_equal(enhanced[0, 2], image[0, 2])
        np.testing.assert_array_equal(enhanced[..., 0], image[..., 0])
        np.testing.assert_array_equal(enhanced[..., 2], image[..., 2])

    def test_does_not_modify_input(self) -> None:
        image = np.array([[[10, 200, 10]]], dtype=np.uint8)
        original = image.copy()

        enhanced = enhance_green(image)

        np.testing.assert_array_equal(image, original)
        self.assertIsNot(enhanced, image)

    def test_validates_shape_dtype_and_strength(self) -> None:
        with self.assertRaises(ValueError):
            enhance_green(np.zeros((2, 2), dtype=np.uint8))
        with self.assertRaises(ValueError):
            enhance_green(np.zeros((2, 2, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            enhance_green(np.zeros((2, 2, 3), dtype=np.uint8), strength=1.1)


if __name__ == "__main__":
    unittest.main()
