from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from plant_counter.audit import audit_annotations


class AnnotationAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.annotations = self.root / "annotations"
        self.dataset = self.root / "dataset"
        self.annotations.mkdir()
        rules = {
            "geometry": {
                "minimum_box_width_pixels": 8,
                "minimum_box_height_pixels": 8,
                "maximum_box_area_ratio": 0.30,
                "duplicate_iou_threshold": 0.95,
            },
            "classes": {
                "wheat": {"coco_id": 1},
                "wild oat": {"coco_id": 2},
                "brome grass": {"coco_id": 3},
                "barley grass": {"coco_id": 4},
            },
        }
        self.rules = self.root / "rules.yaml"
        self.rules.write_text(yaml.safe_dump(rules), encoding="utf-8")
        self.categories = [
            {"id": index + 1, "name": name}
            for index, name in enumerate(rules["classes"])
        ]

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_split(self, split: str, bbox: list[int]) -> None:
        images = self.dataset / "images" / split
        images.mkdir(parents=True)
        filename = f"{split}.jpg"
        (images / filename).write_bytes(b"test")
        payload = {
            "images": [{"id": 1, "file_name": filename, "width": 100, "height": 100}],
            "annotations": [{"id": 2, "image_id": 1, "category_id": 1, "bbox": bbox}],
            "categories": self.categories,
        }
        (self.annotations / f"batch_combined_{split}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_reports_valid_dataset(self) -> None:
        self.write_split("train", [10, 10, 20, 30])
        self.write_split("val", [10, 10, 20, 30])
        report, issues = audit_annotations(self.annotations, self.dataset, self.rules)
        self.assertEqual(report["issues"], {"errors": 0, "warnings": 0})
        self.assertEqual(report["class_instances"]["wheat"], 2)
        self.assertEqual(issues, [])

    def test_flags_outside_and_tiny_box(self) -> None:
        self.write_split("train", [-1, 10, 5, 5])
        self.write_split("val", [10, 10, 20, 30])
        report, issues = audit_annotations(self.annotations, self.dataset, self.rules)
        self.assertEqual(report["issues"]["errors"], 1)
        self.assertTrue(any(item["code"] == "tiny_bbox" for item in issues))

    def test_flags_test_image_copied_from_validation(self) -> None:
        self.write_split("train", [10, 10, 20, 30])
        self.write_split("val", [10, 10, 20, 30])
        test_dir = self.dataset / "images" / "test"
        test_dir.mkdir(parents=True)
        (test_dir / "val.jpg").write_bytes(b"copy")
        report, issues = audit_annotations(self.annotations, self.dataset, self.rules)
        self.assertEqual(report["splits"]["test"]["overlap_with_train_or_val"], 1)
        self.assertTrue(any(item["code"] == "test_overlap_filename" for item in issues))


if __name__ == "__main__":
    unittest.main()
