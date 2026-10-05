"""Build a broadleaf dataset led by CropAndWeed with target-domain replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def make_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external",
        type=Path,
        default=project_root / "data" / "cropandweed_broadleaf",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=project_root / "data" / "21_08_borad_jpg",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "broadleaf_sparse_mixed",
    )
    parser.add_argument(
        "--target-repeats",
        type=int,
        default=16,
        help="Number of target-domain copies in training; 16 gives about 10%% target images.",
    )
    parser.add_argument(
        "--external-limit",
        type=int,
        help="Deterministic external replay subset size; omit to use every external image.",
    )
    return parser


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _target_dirs(root: Path, split: str) -> tuple[Path, Path]:
    return root / split / "images", root / split / "labels"


def _external_dirs(root: Path, split: str) -> tuple[Path, Path]:
    return root / "images" / split, root / "labels" / split


def _add_split(
    image_dir: Path,
    label_dir: Path,
    output: Path,
    split: str,
    prefix: str,
    repeat: int,
    limit: int | None = None,
) -> tuple[int, int, dict[str, int]]:
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if limit is not None:
        if limit < 1:
            raise ValueError("External limit must be positive")
        images.sort(key=lambda path: hashlib.sha256(path.name.encode("utf-8")).digest())
        images = images[:limit]
    image_count = instance_count = 0
    modes = {"hardlink": 0, "copy": 0}
    for repetition in range(repeat):
        repeat_prefix = f"{prefix}r{repetition:02d}__" if repeat > 1 else f"{prefix}__"
        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
            destination_stem = f"{repeat_prefix}{image_path.stem}"
            destination_image = output / "images" / split / f"{destination_stem}{image_path.suffix.lower()}"
            destination_label = output / "labels" / split / f"{destination_stem}.txt"
            if destination_image.exists() or destination_label.exists():
                raise FileExistsError(f"Output already contains {destination_stem}; use a new output folder")
            mode = _link_or_copy(image_path, destination_image)
            modes[mode] += 1
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_path, destination_label)
            image_count += 1
            instance_count += sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return image_count, instance_count, modes


def main() -> None:
    args = make_parser().parse_args()
    if args.target_repeats < 1:
        raise SystemExit("--target-repeats must be at least 1")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output folder is not empty: {output}")

    summary: dict[str, object] = {
        "external": str(args.external.resolve()),
        "target": str(args.target.resolve()),
        "target_repeats": args.target_repeats,
        "external_limit": args.external_limit,
        "splits": {},
    }

    external_images, external_labels = _external_dirs(args.external.resolve(), "train")
    ext_images, ext_instances, ext_modes = _add_split(
        external_images,
        external_labels,
        output,
        "train",
        "external",
        1,
        args.external_limit,
    )
    target_images, target_labels = _target_dirs(args.target.resolve(), "train")
    target_count, target_instances, target_modes = _add_split(
        target_images, target_labels, output, "train", "target", args.target_repeats
    )
    summary["splits"]["train"] = {
        "external_images": ext_images,
        "external_instances": ext_instances,
        "target_images_after_replay": target_count,
        "target_instances_after_replay": target_instances,
        "target_image_fraction": target_count / (ext_images + target_count),
        "materialization": {
            "external": ext_modes,
            "target": target_modes,
        },
    }

    for split in ("val", "test"):
        images, labels = _target_dirs(args.target.resolve(), split)
        image_count, instance_count, modes = _add_split(
            images, labels, output, split, "target", 1
        )
        summary["splits"][split] = {
            "target_images": image_count,
            "target_instances": instance_count,
            "materialization": modes,
        }

    output.mkdir(parents=True, exist_ok=True)
    (output / "data.yaml").write_text(
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
    (output / "mix_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
