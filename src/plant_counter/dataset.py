"""Build a flat YOLO detection dataset from multiple COCO annotation files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SplitReport:
    split: str
    annotation_files: int
    images: int
    labels: int
    instances: int
    empty_images: int
    class_instances: dict[str, int]


class DatasetError(ValueError):
    """Raised when annotations and images cannot form a trustworthy dataset."""


def _read_coco(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Cannot read COCO file {path}: {exc}") from exc

    required = {"images", "annotations", "categories"}
    missing = required.difference(data)
    if missing:
        raise DatasetError(f"{path} is missing COCO fields: {sorted(missing)}")
    return data


def _category_signature(data: dict[str, Any], path: Path) -> tuple[tuple[int, str], ...]:
    signature = tuple(
        sorted((int(item["id"]), str(item["name"]).strip()) for item in data["categories"])
    )
    if not signature or any(not name for _, name in signature):
        raise DatasetError(f"{path} has missing or empty categories")
    if len({category_id for category_id, _ in signature}) != len(signature):
        raise DatasetError(f"{path} contains duplicate category IDs")
    if len({name for _, name in signature}) != len(signature):
        raise DatasetError(f"{path} contains duplicate category names")
    return signature


def _image_index(images_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in images_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.name in index:
            raise DatasetError(f"Duplicate image filename in {images_dir}: {path.name}")
        index[path.name] = path
    return index


def _format_yolo_box(
    bbox: Iterable[Any], width: int, height: int, class_index: int, context: str
) -> str:
    values = list(bbox)
    if len(values) != 4:
        raise DatasetError(f"{context} has an invalid bbox: {values}")
    x, y, box_width, box_height = (float(value) for value in values)
    if width <= 0 or height <= 0 or box_width <= 0 or box_height <= 0:
        raise DatasetError(f"{context} has non-positive image/bbox dimensions")

    x1 = max(0.0, min(x, float(width)))
    y1 = max(0.0, min(y, float(height)))
    x2 = max(0.0, min(x + box_width, float(width)))
    y2 = max(0.0, min(y + box_height, float(height)))
    if x2 <= x1 or y2 <= y1:
        raise DatasetError(f"{context} has a bbox outside the image: {values}")

    center_x = ((x1 + x2) / 2.0) / width
    center_y = ((y1 + y2) / 2.0) / height
    normalized_width = (x2 - x1) / width
    normalized_height = (y2 - y1) / height
    return (
        f"{class_index} {center_x:.6f} {center_y:.6f} "
        f"{normalized_width:.6f} {normalized_height:.6f}"
    )


def build_split(
    annotation_files: list[Path],
    images_dir: Path,
    labels_dir: Path,
    split: str,
    expected_categories: tuple[tuple[int, str], ...] | None = None,
    prune_stale: bool = True,
) -> tuple[SplitReport, tuple[tuple[int, str], ...]]:
    if not annotation_files:
        raise DatasetError(f"No annotation files found for split '{split}'")
    if not images_dir.is_dir():
        raise DatasetError(f"Image directory does not exist: {images_dir}")

    image_index = _image_index(images_dir)
    labels_by_stem: dict[str, list[str]] = {}
    filename_sources: dict[str, Path] = {}
    class_counts: dict[str, int] = defaultdict(int)
    total_instances = 0
    categories = expected_categories

    for annotation_path in annotation_files:
        data = _read_coco(annotation_path)
        current_categories = _category_signature(data, annotation_path)
        if categories is None:
            categories = current_categories
        elif current_categories != categories:
            raise DatasetError(
                f"Category mapping differs in {annotation_path}: "
                f"expected {categories}, got {current_categories}"
            )

        category_to_index = {
            category_id: index for index, (category_id, _) in enumerate(categories)
        }
        category_names = dict(categories)
        images_by_id: dict[int, dict[str, Any]] = {}
        for image in data["images"]:
            image_id = int(image["id"])
            if image_id in images_by_id:
                raise DatasetError(f"Duplicate image ID {image_id} in {annotation_path}")
            images_by_id[image_id] = image

        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in data["annotations"]:
            image_id = int(annotation["image_id"])
            category_id = int(annotation["category_id"])
            if image_id not in images_by_id:
                raise DatasetError(
                    f"Annotation {annotation.get('id')} in {annotation_path} "
                    f"references unknown image ID {image_id}"
                )
            if category_id not in category_to_index:
                raise DatasetError(
                    f"Annotation {annotation.get('id')} in {annotation_path} "
                    f"references unknown category ID {category_id}"
                )
            annotations_by_image[image_id].append(annotation)

        for image_id, image in images_by_id.items():
            filename = Path(str(image["file_name"]).replace("\\", "/")).name
            if filename in filename_sources:
                raise DatasetError(
                    f"Duplicate COCO filename {filename} in {filename_sources[filename]} "
                    f"and {annotation_path}"
                )
            filename_sources[filename] = annotation_path
            if filename not in image_index:
                raise DatasetError(f"Image referenced by COCO is missing: {images_dir / filename}")

            stem = Path(filename).stem
            if stem in labels_by_stem:
                raise DatasetError(f"Duplicate image stem would overwrite a label: {stem}")

            width = int(image["width"])
            height = int(image["height"])
            lines: list[str] = []
            for annotation in annotations_by_image.get(image_id, []):
                category_id = int(annotation["category_id"])
                lines.append(
                    _format_yolo_box(
                        annotation.get("bbox", []),
                        width,
                        height,
                        category_to_index[category_id],
                        f"annotation {annotation.get('id')} in {annotation_path.name}",
                    )
                )
                class_counts[category_names[category_id]] += 1
                total_instances += 1
            labels_by_stem[stem] = lines

    assert categories is not None
    expected_filenames = set(filename_sources)
    actual_filenames = set(image_index)
    extras = sorted(actual_filenames.difference(expected_filenames))
    if extras:
        preview = ", ".join(extras[:5])
        raise DatasetError(
            f"{images_dir} contains {len(extras)} images not referenced by the selected "
            f"annotations (first: {preview})"
        )

    labels_dir.mkdir(parents=True, exist_ok=True)
    expected_label_names = {f"{stem}.txt" for stem in labels_by_stem}
    if prune_stale:
        for stale_path in labels_dir.glob("*.txt"):
            if stale_path.name not in expected_label_names:
                stale_path.unlink()

    for stem, lines in labels_by_stem.items():
        target = labels_dir / f"{stem}.txt"
        content = "\n".join(lines)
        if content:
            content += "\n"
        target.write_text(content, encoding="utf-8")

    report = SplitReport(
        split=split,
        annotation_files=len(annotation_files),
        images=len(labels_by_stem),
        labels=len(list(labels_dir.glob("*.txt"))),
        instances=total_instances,
        empty_images=sum(not lines for lines in labels_by_stem.values()),
        class_instances={name: class_counts.get(name, 0) for _, name in categories},
    )
    if report.images != report.labels:
        raise DatasetError(
            f"Label count mismatch after build: {report.images} images vs {report.labels} labels"
        )
    return report, categories


def build_dataset(
    annotations_dir: Path,
    dataset_dir: Path,
    yaml_path: Path,
    report_path: Path,
    prune_stale: bool = True,
) -> dict[str, Any]:
    reports: list[SplitReport] = []
    categories: tuple[tuple[int, str], ...] | None = None
    for split in ("train", "val"):
        files = sorted(annotations_dir.glob(f"*_combined_{split}.json"))
        report, categories = build_split(
            files,
            dataset_dir / "images" / split,
            dataset_dir / "labels" / split,
            split,
            expected_categories=categories,
            prune_stale=prune_stale,
        )
        reports.append(report)

    assert categories is not None
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_lines = [
        f"path: {dataset_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: '{name.replace(chr(39), chr(39) * 2)}'" for index, (_, name) in enumerate(categories))
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    output = {
        "dataset_dir": str(dataset_dir.resolve()),
        "yaml": str(yaml_path.resolve()),
        "categories": [name for _, name in categories],
        "splits": [asdict(report) for report in reports],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and strictly validate YOLO labels from all COCO train/val files."
    )
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--yaml", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--no-prune-stale",
        action="store_true",
        help="Keep label files that are not present in the selected annotations.",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    try:
        report = build_dataset(
            args.annotations_dir.resolve(),
            args.dataset_dir.resolve(),
            args.yaml.resolve(),
            args.report.resolve(),
            prune_stale=not args.no_prune_stale,
        )
    except DatasetError as exc:
        raise SystemExit(f"Dataset build failed: {exc}") from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
