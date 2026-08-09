"""Systematically check counting accuracy across the whole validation set.

For every image in data/yolo/images/val, compares the model's predicted
crop/weed counts against the real ground-truth counts (from the label .txt
files), and reports aggregate accuracy - not just box-detection metrics
like mAP, but the thing that actually matters for this project: how close
are the predicted counts to reality.

Usage:
    python scripts/evaluate_counts.py --weights runs/colab_hires/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

IMAGES_DIR = Path("data/yolo/images/val")
LABELS_DIR = Path("data/yolo/labels/val")
CLASS_NAMES = ["crop", "weed"]


def ground_truth_counts(label_path: Path) -> dict:
    counts = {"crop": 0, "weed": 0}
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            class_id = int(line.split()[0])
            counts[CLASS_NAMES[class_id]] += 1
    return counts


def predicted_counts(result) -> dict:
    counts = {"crop": 0, "weed": 0}
    for box in result.boxes:
        class_id = int(box.cls.item())
        counts[CLASS_NAMES[class_id]] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.3, help="NMS IoU threshold - lower merges nearby boxes more aggressively")
    args = parser.parse_args()

    model = YOLO(args.weights)
    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))

    rows = []
    for image_path in image_paths:
        label_path = LABELS_DIR / (image_path.stem + ".txt")
        truth = ground_truth_counts(label_path)

        result = model.predict(source=str(image_path), conf=args.conf, iou=args.iou, verbose=False)[0]
        pred = predicted_counts(result)

        rows.append({
            "file": image_path.name,
            "crop_true": truth["crop"], "crop_pred": pred["crop"],
            "weed_true": truth["weed"], "weed_pred": pred["weed"],
        })

    print(f"Evaluated {len(rows)} validation images\n")

    for cls in ["crop", "weed"]:
        errors = [r[f"{cls}_pred"] - r[f"{cls}_true"] for r in rows]
        abs_errors = [abs(e) for e in errors]
        exact = sum(1 for e in errors if e == 0)
        within_1 = sum(1 for e in abs_errors if e <= 1)
        within_2 = sum(1 for e in abs_errors if e <= 2)
        mae = sum(abs_errors) / len(abs_errors)
        bias = sum(errors) / len(errors)

        print(f"--- {cls} ---")
        print(f"  Mean absolute error: {mae:.2f} plants/image")
        print(f"  Mean bias:           {bias:+.2f} (positive = overcounts on average)")
        print(f"  Exact count match:   {exact}/{len(rows)} ({100*exact/len(rows):.0f}%)")
        print(f"  Within 1 plant:      {within_1}/{len(rows)} ({100*within_1/len(rows):.0f}%)")
        print(f"  Within 2 plants:     {within_2}/{len(rows)} ({100*within_2/len(rows):.0f}%)")
        print()

    worst = sorted(rows, key=lambda r: abs(r["crop_pred"] - r["crop_true"]) + abs(r["weed_pred"] - r["weed_true"]), reverse=True)[:5]
    print("--- 5 worst images (by total count error) ---")
    for r in worst:
        print(f"  {r['file']}: crop true={r['crop_true']} pred={r['crop_pred']}, weed true={r['weed_true']} pred={r['weed_pred']}")


if __name__ == "__main__":
    main()
