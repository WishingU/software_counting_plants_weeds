"""Audit COCO annotations against the project's annotation contract."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _issue(
    issues: list[dict[str, Any]], severity: str, code: str, source: Path, message: str,
    image: str = "", annotation_id: object = "",
) -> None:
    issues.append(
        {
            "severity": severity,
            "code": code,
            "source": source.name,
            "image": image,
            "annotation_id": annotation_id,
            "message": message,
        }
    )


def _iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    intersection_width = max(0.0, min(lx + lw, rx + rw) - max(lx, rx))
    intersection_height = max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    intersection = intersection_width * intersection_height
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def audit_annotations(
    annotations_dir: Path,
    dataset_dir: Path,
    rules_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load annotation rules") from exc

    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    class_rules = rules["classes"]
    expected_categories = {
        int(rule["coco_id"]): name for name, rule in class_rules.items()
    }
    geometry = rules["geometry"]
    min_width = float(geometry["minimum_box_width_pixels"])
    min_height = float(geometry["minimum_box_height_pixels"])
    max_area_ratio = float(geometry["maximum_box_area_ratio"])
    duplicate_threshold = float(geometry["duplicate_iou_threshold"])

    issues: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    split_counts: dict[str, dict[str, int]] = {}
    widths: list[float] = []
    heights: list[float] = []
    area_ratios: list[float] = []
    total_files = 0
    all_filenames: dict[tuple[str, str], Path] = {}
    annotated_filename_splits: dict[str, str] = {}

    for split in ("train", "val"):
        split_images = 0
        split_instances = 0
        files = sorted(annotations_dir.glob(f"*_combined_{split}.json"))
        if not files:
            _issue(issues, "error", "missing_split", annotations_dir, f"No {split} COCO files")
        for source in files:
            total_files += 1
            try:
                data = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                _issue(issues, "error", "invalid_json", source, str(exc))
                continue

            actual_categories = {
                int(item["id"]): str(item["name"]).strip() for item in data.get("categories", [])
            }
            if actual_categories != expected_categories:
                _issue(
                    issues, "error", "category_contract", source,
                    f"Expected {expected_categories}, found {actual_categories}",
                )

            images: dict[int, dict[str, Any]] = {}
            for image in data.get("images", []):
                image_id = int(image["id"])
                filename = Path(str(image["file_name"]).replace("\\", "/")).name
                if image_id in images:
                    _issue(issues, "error", "duplicate_image_id", source, f"Image ID {image_id}", filename)
                    continue
                filename_key = (split, filename)
                if filename_key in all_filenames:
                    _issue(
                        issues, "error", "duplicate_filename", source,
                        f"Also present in {all_filenames[filename_key].name}", filename,
                    )
                all_filenames[filename_key] = source
                previous_split = annotated_filename_splits.get(filename)
                if previous_split is not None and previous_split != split:
                    _issue(
                        issues, "error", "split_leakage_filename", source,
                        f"Filename also occurs in {previous_split}; train/val must be disjoint",
                        filename,
                    )
                annotated_filename_splits[filename] = split
                images[image_id] = image
                split_images += 1
                if not (dataset_dir / "images" / split / filename).is_file():
                    _issue(issues, "error", "missing_image", source, "Referenced image is absent", filename)

            seen_annotation_ids: set[int] = set()
            boxes_by_image: dict[int, list[tuple[int, int, tuple[float, ...], str]]] = defaultdict(list)
            for annotation in data.get("annotations", []):
                annotation_id = int(annotation["id"])
                image_id = int(annotation["image_id"])
                category_id = int(annotation["category_id"])
                if annotation_id in seen_annotation_ids:
                    _issue(issues, "error", "duplicate_annotation_id", source, "Duplicate annotation ID", annotation_id=annotation_id)
                    continue
                seen_annotation_ids.add(annotation_id)
                if image_id not in images:
                    _issue(issues, "error", "unknown_image", source, f"Unknown image ID {image_id}", annotation_id=annotation_id)
                    continue
                image = images[image_id]
                filename = Path(str(image["file_name"]).replace("\\", "/")).name
                if category_id not in expected_categories:
                    _issue(issues, "error", "unknown_category", source, f"Category ID {category_id}", filename, annotation_id)
                    continue
                try:
                    x, y, width, height = (float(value) for value in annotation["bbox"])
                except (KeyError, TypeError, ValueError):
                    _issue(issues, "error", "invalid_bbox", source, "BBox must contain four numbers", filename, annotation_id)
                    continue
                image_width, image_height = int(image["width"]), int(image["height"])
                if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
                    _issue(issues, "error", "non_positive_bbox", source, "Non-positive bbox or image size", filename, annotation_id)
                    continue
                if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
                    _issue(issues, "error", "bbox_outside_image", source, "BBox crosses the image boundary", filename, annotation_id)
                if width < min_width or height < min_height:
                    _issue(issues, "warning", "tiny_bbox", source, f"BBox is {width:.1f} x {height:.1f} px", filename, annotation_id)
                ratio = width * height / (image_width * image_height)
                if ratio > max_area_ratio:
                    _issue(issues, "warning", "large_bbox", source, f"BBox occupies {ratio:.1%} of image", filename, annotation_id)
                class_name = expected_categories[category_id]
                class_counts[class_name] += 1
                split_instances += 1
                widths.append(width)
                heights.append(height)
                area_ratios.append(ratio)
                boxes_by_image[image_id].append((annotation_id, category_id, (x, y, width, height), filename))

            for candidates in boxes_by_image.values():
                for index, left in enumerate(candidates):
                    for right in candidates[index + 1 :]:
                        if left[1] == right[1] and _iou(left[2], right[2]) >= duplicate_threshold:
                            _issue(
                                issues, "warning", "possible_duplicate", source,
                                f"Same-class boxes {left[0]} and {right[0]} have IoU >= {duplicate_threshold}",
                                left[3], f"{left[0]},{right[0]}",
                            )
        split_counts[split] = {"images": split_images, "instances": split_instances}

    test_images_dir = dataset_dir / "images" / "test"
    if test_images_dir.is_dir():
        test_files = [path for path in test_images_dir.iterdir() if path.is_file()]
        test_overlap = 0
        for path in test_files:
            previous_split = annotated_filename_splits.get(path.name)
            if previous_split is not None:
                test_overlap += 1
                _issue(
                    issues, "warning", "test_overlap_filename", annotations_dir,
                    f"Test image is also in {previous_split}; do not report it as an independent test",
                    path.name,
                )
        split_counts["test"] = {
            "images": len(test_files),
            "instances": 0,
            "overlap_with_train_or_val": test_overlap,
        }

    severities = Counter(item["severity"] for item in issues)
    report: dict[str, Any] = {
        "rules": str(rules_path.resolve()),
        "annotation_files": total_files,
        "splits": split_counts,
        "class_instances": {name: class_counts[name] for name in class_rules},
        "bbox_pixels": {
            "width_p05": _percentile(widths, 0.05),
            "width_median": _percentile(widths, 0.50),
            "height_p05": _percentile(heights, 0.05),
            "height_median": _percentile(heights, 0.50),
            "area_ratio_p95": _percentile(area_ratios, 0.95),
        },
        "issues": {"errors": severities["error"], "warnings": severities["warning"]},
    }
    return report, issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit COCO annotations against project rules.")
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, issues = audit_annotations(
        args.annotations_dir.resolve(), args.dataset_dir.resolve(), args.rules.resolve()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "annotation_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "annotation_issues.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["severity", "code", "source", "image", "annotation_id", "message"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(issues)
    print(json.dumps(report, indent=2))
    if report["issues"]["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
