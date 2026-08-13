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

Default conf=0.35 was chosen by sweeping thresholds against all 301
validation images and picking the best precision/recall balance (F1).
See scripts/evaluate_plant_detection.py for how this was measured.
"""

import argparse

from ultralytics import YOLO

DEFAULT_WEIGHTS = "runs/colab_50epoch/best.pt"
CLASS_NAMES = ["crop", "weed"]


def count_plants(image_path: str, weights_path: str, confidence: float = 0.35, iou: float = 0.3):
    model = YOLO(weights_path)
    results = model.predict(source=image_path, conf=confidence, iou=iou, verbose=False)
    result = results[0]

    counts = {name: 0 for name in CLASS_NAMES}
    for box in result.boxes:
        class_id = int(box.cls.item())
        counts[CLASS_NAMES[class_id]] += 1

    return counts, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to the image to count plants in")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to trained model weights")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (0-1)")
    parser.add_argument("--iou", type=float, default=0.3, help="NMS IoU threshold - lower merges nearby boxes more aggressively")
    parser.add_argument("--save", help="Optional path to save the annotated image")
    args = parser.parse_args()

    counts, result = count_plants(args.image, args.weights, args.conf, args.iou)

    total = sum(counts.values())
    print(f"\n{args.image}")
    for name, n in counts.items():
        print(f"  {name}: {n}")
    print(f"  total: {total}")

    if args.save:
        result.save(filename=args.save)
        print(f"\nAnnotated image saved to {args.save}")


if __name__ == "__main__":
    main()
