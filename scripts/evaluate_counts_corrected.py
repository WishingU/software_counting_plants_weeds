"""Estimate counting accuracy after correcting for known annotation gaps.

The systematic audits (audit_false_positives.py, audit_threshold_gap.py)
found 70/70 sampled "false positives" were actually real, correctly-detected
plants missing from the dataset's ground-truth labels. Using the "rule of
three" for small-sample zero-event estimates, that bounds the true error
rate among those flagged detections at roughly <=4.3% (95% confidence).

This script re-scores counting accuracy using that correction: when the
model's predicted count exceeds the true (labeled) count for an image, we
treat ~95.7% of that excess as legitimate (real, unlabeled plants) rather
than error. Undercounting (predicted < true) is left unadjusted, since the
audits didn't test that direction - it's a separate, unverified failure
mode (e.g. dense overlap cases).

This is an ESTIMATE, not a re-measurement against corrected ground truth -
it approximates what accuracy would look like if the labels were complete,
based on the confirmed real-detection rate from the audits.

Usage:
    python scripts/evaluate_counts_corrected.py --weights runs/colab_50epoch/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

IMAGES_DIR = Path("data/yolo/images/val")
LABELS_DIR = Path("data/yolo/labels/val")
CLASS_NAMES = ["crop", "weed"]
REAL_DETECTION_RATE = 0.957  # from 70/70 audit samples, rule-of-three lower bound


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
    parser.add_argument("--iou", type=float, default=0.3)
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
            "total_true": truth["crop"] + truth["weed"],
            "total_pred": pred["crop"] + pred["weed"],
        })

    print(f"Evaluated {len(rows)} validation images\n")
    print(f"Applying correction: {REAL_DETECTION_RATE*100:.1f}% of overcounting excess treated as real, unlabeled plants\n")

    raw_errors, corrected_errors = [], []
    for r in rows:
        true, pred = r["total_true"], r["total_pred"]
        raw_errors.append(abs(pred - true))
        if pred > true:
            excess = pred - true
            corrected_errors.append(excess * (1 - REAL_DETECTION_RATE))
        else:
            corrected_errors.append(true - pred)

    def summarize(errors, label):
        mae = sum(errors) / len(errors)
        within_1 = sum(1 for e in errors if e <= 1)
        within_2 = sum(1 for e in errors if e <= 2)
        print(f"--- {label} ---")
        print(f"  Mean absolute error: {mae:.2f} plants/image")
        print(f"  Within 1 plant:      {within_1}/{len(errors)} ({100*within_1/len(errors):.0f}%)")
        print(f"  Within 2 plants:     {within_2}/{len(errors)} ({100*within_2/len(errors):.0f}%)")
        print()

    summarize(raw_errors, "Raw (against incomplete labels)")
    summarize(corrected_errors, "Corrected estimate (accounting for label gaps)")


if __name__ == "__main__":
    main()
