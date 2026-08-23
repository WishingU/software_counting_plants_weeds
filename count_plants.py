"""Count crop and weed plants in an image using our trained YOLOv8 model.

Usage:
    python count_plants.py path/to/image.jpg
    python count_plants.py path/to/image.jpg --weights runs/colab_hires/best.pt
    python count_plants.py path/to/image.jpg --save annotated.jpg

Runs entirely on CPU/MPS locally - no GPU needed for inference, only for
training. Prints the crop and weed counts, and optionally saves a copy of
the image with the detected boxes drawn on it.

Default iou=0.3 is lower than YOLO's usual default (0.7) - validated against
all 301 validation images to meaningfully reduce duplicate-box overcounting
on long, thin, curving grass blades without losing genuinely separate nearby
plants. See scripts/evaluate_counts.py for how this was measured.

Default conf=0.25: an earlier pass raised this to 0.35 to improve measured
precision, but a systematic audit (scripts/audit_false_positives.py and
scripts/audit_threshold_gap.py) found that ~100% of a 70-sample check of
"false positives" - including ones only caught by the 0.25 threshold - were
actually real, correctly-detected plants missing from the dataset's
ground-truth labels. Raising the threshold was filtering out genuine
detections, not noise, and made real counting accuracy worse. Reverted to
0.25 for that reason.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "colab_50epoch" / "best.pt"


def count_plants(
    image_path: Path,
    weights_path: Path,
    confidence: float = 0.25,
    iou: float = 0.3,
    device: str | None = None,
):
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights do not exist: {weights_path}")
    model = YOLO(str(weights_path))
    results = model.predict(
        source=str(image_path), conf=confidence, iou=iou, device=device, verbose=False
    )
    result = results[0]

    names = {int(index): str(name) for index, name in model.names.items()}
    counts = {name: 0 for name in names.values()}
    for box in result.boxes:
        class_id = int(box.cls.item())
        counts[names[class_id]] += 1

    return counts, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Path to the image to count plants in")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to trained model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0-1)")
    parser.add_argument("--iou", type=float, default=0.3, help="NMS IoU threshold - lower merges nearby boxes more aggressively")
    parser.add_argument("--device", help="CUDA device such as 0, or cpu (default: auto)")
    parser.add_argument("--save", type=Path, help="Optional path to save the annotated image")
    args = parser.parse_args()

    try:
        counts, result = count_plants(args.image.resolve(), args.weights.resolve(), args.conf, args.iou, args.device)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    total = sum(counts.values())
    print(f"\n{args.image}")
    for name, n in counts.items():
        print(f"  {name}: {n}")
    print(f"  total: {total}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        result.save(filename=str(args.save))
        print(f"\nAnnotated image saved to {args.save}")


if __name__ == "__main__":
    main()
