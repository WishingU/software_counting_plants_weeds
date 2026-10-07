from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.split_images import split_image


class SplitImageTests(unittest.TestCase):
    def test_grid_preserves_all_edge_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            output = root / "parts"
            image = np.zeros((5, 7, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(source), image))

            parts = split_image(source, output, grid=2)

            self.assertEqual(len(parts), 4)
            shapes = [cv2.imread(str(path)).shape[:2] for path in parts]
            self.assertEqual(shapes, [(2, 3), (2, 4), (3, 3), (3, 4)])
            self.assertEqual(sum(height * width for height, width in shapes), 35)

    def test_rejects_grid_larger_than_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "tiny.png"
            self.assertTrue(
                cv2.imwrite(str(source), np.zeros((2, 2, 3), dtype=np.uint8))
            )

            with self.assertRaises(ValueError):
                split_image(source, root / "parts", grid=3)


if __name__ == "__main__":
    unittest.main()
