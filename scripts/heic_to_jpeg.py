"""Convert HEIC images in a directory to JPEG.

Example:
    python scripts/heic_to_jpeg.py data/heic data/jpeg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pillow_heif
from PIL import Image


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert HEIC images to JPEG.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--quality", type=int, default=95, choices=range(1, 101))
    return parser


def main() -> None:
    args = make_parser().parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in {".heic", ".heif"}
    )
    if not image_paths:
        raise SystemExit(f"No HEIC or HEIF images found in {input_dir}")

    pillow_heif.register_heif_opener()
    output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for image_path in image_paths:
        output_path = output_dir / f"{image_path.stem}.jpg"
        try:
            with Image.open(image_path) as image:
                image.convert("RGB").save(output_path, quality=args.quality)
            print(f"{image_path.name} -> {output_path.name}")
        except (OSError, ValueError) as exc:
            failures += 1
            print(f"{image_path.name}: {exc}")

    converted = len(image_paths) - failures
    print(f"Converted {converted}/{len(image_paths)} images into {output_dir}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
