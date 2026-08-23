"""Rebuild the portable two-class YOLO dataset from COCO annotations."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_MAP = {
    "wheat": 0,
    "wild oat": 1,
    "brome grass": 1,
    "barley grass": 1,
}
CLASS_NAMES = {0: "crop", 1: "weed"}


class BuildError(RuntimeError):
    """Raised when the source data cannot be rebuilt safely."""


def image_index(source: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.name in result:
            raise BuildError(f"Duplicate image filename: {path.name}")
        result[path.name] = path
    return result


def output_split(annotation_path: Path) -> str:
    stem = annotation_path.stem.lower()
    for split in ("train", "val", "test"):
        if stem.endswith(f"_combined_{split}"):
            return split
    raise BuildError(f"Cannot determine split from {annotation_path.name}")


def format_box(bbox: list[float], width: int, height: int, context: str) -> str:
    if len(bbox) != 4 or width <= 0 or height <= 0:
        raise BuildError(f"Invalid box or image size in {context}")
    x, y, box_width, box_height = (float(value) for value in bbox)
    if box_width <= 0 or box_height <= 0:
        raise BuildError(f"Non-positive box in {context}")
    if x < 0 or y < 0 or x + box_width > width or y + box_height > height:
        raise BuildError(f"Box crosses image boundary in {context}")
    center_x = (x + box_width / 2) / width
    center_y = (y + box_height / 2) / height
    return f"{center_x:.6f} {center_y:.6f} {box_width / width:.6f} {box_height / height:.6f}"


def materialize_image(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def build(annotations_dir: Path, source_images: Path, output: Path) -> dict[str, object]:
    annotation_paths = sorted(annotations_dir.glob("*_combined_*.json"))
    if not annotation_paths:
        raise BuildError(f"No COCO annotation files found in {annotations_dir}")

    images = image_index(source_images)
    staging = output.parent / f"{output.name}_rebuild_staging"
    if staging.exists():
        raise BuildError(f"Staging directory already exists: {staging}")
    if (output / "images").exists() or (output / "labels").exists():
        raise BuildError(f"Output already contains images or labels: {output}")

    split_images: dict[str, set[str]] = {name: set() for name in ("train", "val", "test")}
    split_instances: dict[str, Counter[int]] = {
        name: Counter() for name in ("train", "val", "test")
    }
    materialized: Counter[str] = Counter()

    try:
        for annotation_path in annotation_paths:
            split = output_split(annotation_path)
            data = json.loads(annotation_path.read_text(encoding="utf-8"))
            categories = {int(item["id"]): str(item["name"]).strip() for item in data["categories"]}
            unknown = set(categories.values()) - set(CLASS_MAP)
            if unknown:
                raise BuildError(f"Unknown categories in {annotation_path.name}: {sorted(unknown)}")

            annotations_by_image: dict[int, list[dict[str, object]]] = defaultdict(list)
            for annotation in data["annotations"]:
                annotations_by_image[int(annotation["image_id"])].append(annotation)

            for image in data["images"]:
                image_id = int(image["id"])
                filename = Path(str(image["file_name"]).replace("\\", "/")).name
                if filename in split_images[split]:
                    raise BuildError(f"Duplicate {split} image in annotations: {filename}")
                if filename not in images:
                    raise BuildError(f"Annotated image is missing: {filename}")
                split_images[split].add(filename)

                target_image = staging / "images" / split / filename
                target_label = staging / "labels" / split / f"{Path(filename).stem}.txt"
                target_image.parent.mkdir(parents=True, exist_ok=True)
                target_label.parent.mkdir(parents=True, exist_ok=True)
                materialized[materialize_image(images[filename], target_image)] += 1

                lines: list[str] = []
                for annotation in annotations_by_image.get(image_id, []):
                    category_name = categories[int(annotation["category_id"])]
                    class_id = CLASS_MAP[category_name]
                    coordinates = format_box(
                        annotation["bbox"],
                        int(image["width"]),
                        int(image["height"]),
                        f"annotation {annotation.get('id')} in {annotation_path.name}",
                    )
                    lines.append(f"{class_id} {coordinates}")
                    split_instances[split][class_id] += 1
                target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        used = set().union(*split_images.values())
        if used != set(images):
            extras = sorted(set(images) - used)
            raise BuildError(f"Source contains {len(extras)} unannotated images; first: {extras[:5]}")
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = split_images[left] & split_images[right]
            if overlap:
                raise BuildError(f"Split leakage between {left} and {right}: {sorted(overlap)[:5]}")

        output.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging / "images"), str(output / "images"))
        shutil.move(str(staging / "labels"), str(output / "labels"))
        (output / "data.yaml").write_text(
            f"path: {output.resolve().as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n\n"
            "names:\n"
            "  0: crop\n"
            "  1: weed\n",
            encoding="utf-8",
        )
        report = {
            "source_annotations": str(annotations_dir.resolve()),
            "source_images": str(source_images.resolve()),
            "output": str(output.resolve()),
            "classes": CLASS_NAMES,
            "image_materialization": dict(materialized),
            "splits": {
                split: {
                    "images": len(split_images[split]),
                    "labels": len(split_images[split]),
                    "instances": sum(split_instances[split].values()),
                    "class_instances": {
                        CLASS_NAMES[index]: split_instances[split][index] for index in CLASS_NAMES
                    },
                }
                for split in ("train", "val", "test")
            },
        }
        (output / "build_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--source-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build(args.annotations.resolve(), args.source_images.resolve(), args.output.resolve())
    except BuildError as exc:
        raise SystemExit(f"Dataset rebuild failed: {exc}") from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
