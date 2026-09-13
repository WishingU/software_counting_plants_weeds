"""Split matching images in a directory into an equal square grid.

Examples:
    python scripts/split_images.py data/source data/split --grid 3
    python scripts/split_images.py data/source data/wild_split --grid 2 --name-contains wild
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def split_image(image_path: Path, output_dir: Path, grid: int) -> list[Path]:
    """Split one image into grid x grid parts without dropping edge pixels."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    height, width = image.shape[:2]
    if grid < 1 or grid > min(height, width):
        raise ValueError(
            f"Grid must be between 1 and the shortest image dimension "
            f"({min(height, width)}), got {grid}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    part = 1
    for row in range(grid):
        y1 = row * height // grid
        y2 = (row + 1) * height // grid
        for column in range(grid):
            x1 = column * width // grid
            x2 = (column + 1) * width // grid
            output_path = output_dir / f"{image_path.stem}_part{part}.jpg"
            if not cv2.imwrite(str(output_path), image[y1:y2, x1:x2]):
                raise OSError(f"Unable to write image: {output_path}")
            written.append(output_path)
            part += 1
    return written


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split JPG and PNG images into an equal square grid."
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--grid",
        type=int,
        default=3,
        help="Rows and columns; 2 creates 4 parts and 3 creates 9.",
    )
    parser.add_argument(
        "--name-contains",
        default="",
        help="Only process filenames containing this text (case-insensitive).",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if args.grid < 1:
        raise SystemExit("--grid must be at least 1")

    needle = args.name_contains.casefold()
    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.casefold() in IMAGE_EXTENSIONS
        and needle in path.stem.casefold()
    )
    if not image_paths:
        raise SystemExit(f"No matching JPG or PNG images found in {input_dir}")

    failures = 0
    for image_path in image_paths:
        try:
            written = split_image(image_path, output_dir, args.grid)
            print(f"{image_path.name}: wrote {len(written)} parts")
        except (OSError, ValueError) as exc:
            failures += 1
            print(f"{image_path.name}: {exc}")

    processed = len(image_paths) - failures
    print(f"Processed {processed}/{len(image_paths)} images into {output_dir}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
