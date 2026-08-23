from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from plant_counter.dataset import DatasetError, build_dataset


class DatasetBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.annotations = self.root / "annotations"
        self.dataset = self.root / "dataset"
        self.annotations.mkdir()
        for split in ("train", "val"):
            (self.dataset / "images" / split).mkdir(parents=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_split(self, split: str, filename: str, category_name: str = "wheat") -> None:
        (self.dataset / "images" / split / filename).write_bytes(b"image")
        payload = {
            "images": [{"id": 10, "file_name": filename, "width": 100, "height": 50}],
            "annotations": [
                {"id": 20, "image_id": 10, "category_id": 1, "bbox": [10, 5, 20, 10]}
            ],
            "categories": [{"id": 1, "name": category_name}],
        }
        (self.annotations / f"batch_combined_{split}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_builds_one_label_per_image_and_normalizes_box(self) -> None:
        self.write_split("train", "train.jpg")
        self.write_split("val", "val.jpg")
        report = build_dataset(
            self.annotations,
            self.dataset,
            self.root / "grass.yaml",
            self.root / "report.json",
        )
        self.assertEqual(report["splits"][0]["images"], 1)
        self.assertEqual(
            (self.dataset / "labels" / "train" / "train.txt").read_text().strip(),
            "0 0.200000 0.200000 0.200000 0.200000",
        )

    def test_rejects_category_mapping_drift(self) -> None:
        self.write_split("train", "train.jpg", "wheat")
        self.write_split("val", "val.jpg", "not-wheat")
        with self.assertRaises(DatasetError):
            build_dataset(
                self.annotations,
                self.dataset,
                self.root / "grass.yaml",
                self.root / "report.json",
            )


if __name__ == "__main__":
    unittest.main()
