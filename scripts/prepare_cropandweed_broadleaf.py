"""Convert CropAndWeed bounding boxes to one-class YOLO detection data.

The target class is defined as the official CropAndWeed ``Weed`` mapping with
all monocot grass categories removed. Crop categories and grass weeds remain
in the selected images as hard negatives. Images are split by acquisition
session so near-duplicate frames cannot cross dataset splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "external" / "cropandweed-dataset" / "data_raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "cropandweed_broadleaf"

# Official CropAndWeed IDs mapped to Weed, excluding the grass IDs documented
# in the Fine24 ``Grasses`` group.
BROADLEAF_WEED_CLASSES = {
    22: "Poppy",
    29: "Hybrid goosefoot",
    30: "Black-bindweed",
    32: "Red-root amaranth",
    33: "White goosefoot",
    34: "Thorn apple",
    35: "Potato weed",
    36: "German chamomile",
    37: "Saltbush",
    38: "Creeping thistle",
    39: "Field milk thistle",
    41: "Black nightshade",
    42: "Mercuries",
    44: "Pale persicaria",
    45: "Geraniums",
    47: "Whitetop",
    49: "Frosted orach",
    50: "Black horehound",
    51: "Shepherds purse",
    52: "Field bindweed",
    54: "Hedge mustard",
    56: "Speedwell",
    57: "Broadleaf plantain",
    58: "White ball-mustard",
    59: "Peppermint",
    60: "Field pennycress",
    61: "Corn spurry",
    63: "Common fumitory",
    64: "Ivy-leaved speedwell",
    66: "Redshank",
    67: "Common hemp-nettle",
    70: "Small geranium",
    71: "Cornflower",
    72: "Common corn-cockle",
    76: "Purple dead-nettle",
    77: "Ribwort plantain",
    78: "Pineappleweed",
    79: "Common chickweed",
    80: "Hedge mustard",
    83: "Yellow rocket",
    85: "Red poppy",
    87: "Knotgrass",
    88: "Prickly lettuce",
    89: "Copse-bindweed",
    91: "Common buckwheat",
    96: "Field mustard",
}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    return parser


def session_key(stem: str) -> str:
    parts = stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"Unexpected CropAndWeed filename: {stem}")
    return "-".join(parts[:2])


def choose_split(session: str, seed: int, train_fraction: float, val_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{session}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < train_fraction:
        return "train"
    if value < train_fraction + val_fraction:
        return "val"
    return "test"


def link_or_copy(source: Path, destination: Path) -> str:
    if destination.exists():
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def convert_box(row: list[str], width: int, height: int) -> tuple[int, str] | None:
    if len(row) < 5:
        raise ValueError(f"Expected at least five CSV columns, received {row!r}")
    left, top, right, bottom = (float(value) for value in row[:4])
    source_class = int(row[4])
    if source_class not in BROADLEAF_WEED_CLASSES:
        return None
    left = min(max(left, 0.0), float(width))
    right = min(max(right, 0.0), float(width))
    top = min(max(top, 0.0), float(height))
    bottom = min(max(bottom, 0.0), float(height))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid box after clipping: {row!r}")
    x_center = ((left + right) / 2.0) / width
    y_center = ((top + bottom) / 2.0) / height
    box_width = (right - left) / width
    box_height = (bottom - top) / height
    label = f"0 {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"
    return source_class, label


def main() -> None:
    args = make_parser().parse_args()
    if args.train_fraction <= 0 or args.val_fraction <= 0:
        raise SystemExit("Train and validation fractions must be positive.")
    if args.train_fraction + args.val_fraction >= 1:
        raise SystemExit("Train and validation fractions must leave a positive test fraction.")

    source = args.source.resolve()
    output = args.output.resolve()
    image_root = source / "images"
    bbox_root = source / "bboxes" / "CropAndWeed"
    if not image_root.is_dir() or not bbox_root.is_dir():
        raise SystemExit(f"CropAndWeed images or bounding boxes are missing under {source}")

    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_counts: Counter[str] = Counter()
    positive_image_counts: Counter[str] = Counter()
    instance_counts: Counter[str] = Counter()
    source_class_instances: Counter[int] = Counter()
    source_class_images: Counter[int] = Counter()
    link_modes: Counter[str] = Counter()
    manifest_rows: list[list[str | int]] = []
    sessions_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

    image_paths = sorted(image_root.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"No JPG images found under {image_root}")

    for image_path in image_paths:
        stem = image_path.stem
        csv_path = bbox_root / f"{stem}.csv"
        if not csv_path.is_file():
            raise SystemExit(f"Missing bounding-box file for {image_path.name}: {csv_path}")
        session = session_key(stem)
        split = choose_split(session, args.seed, args.train_fraction, args.val_fraction)
        sessions_by_split[split].add(session)

        with Image.open(image_path) as image:
            width, height = image.size

        converted_pairs: list[tuple[int, str]] = []
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if not row:
                    continue
                converted = convert_box(row, width, height)
                if converted is None:
                    continue
                converted_pairs.append(converted)

        # The official release contains at least one exact duplicate row.
        # Preserve order while ensuring each box appears once per image.
        converted_pairs = list(dict.fromkeys(converted_pairs))
        labels = [label for _, label in converted_pairs]
        image_source_classes = {source_class for source_class, _ in converted_pairs}
        for source_class, _ in converted_pairs:
            source_class_instances[source_class] += 1

        for source_class in image_source_classes:
            source_class_images[source_class] += 1

        destination_image = output / "images" / split / image_path.name
        destination_label = output / "labels" / split / f"{stem}.txt"
        link_modes[link_or_copy(image_path, destination_image)] += 1
        destination_label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")

        image_counts[split] += 1
        instance_counts[split] += len(labels)
        if labels:
            positive_image_counts[split] += 1
        manifest_rows.append([image_path.name, session, split, len(labels)])

    yaml_path = output / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: broadleaf_weed",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with (output / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "session", "split", "broadleaf_instances"])
        writer.writerows(manifest_rows)

    summary = {
        "source": str(source),
        "output": str(output),
        "seed": args.seed,
        "split_strategy": "SHA-256 deterministic acquisition-session split",
        "mapped_classes": {
            str(class_id): {
                "name": name,
                "instances": source_class_instances[class_id],
                "images": source_class_images[class_id],
            }
            for class_id, name in BROADLEAF_WEED_CLASSES.items()
        },
        "images": dict(image_counts),
        "positive_images": dict(positive_image_counts),
        "instances": dict(instance_counts),
        "sessions": {split: len(values) for split, values in sessions_by_split.items()},
        "image_materialization": dict(link_modes),
    }
    (output / "mapping_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
