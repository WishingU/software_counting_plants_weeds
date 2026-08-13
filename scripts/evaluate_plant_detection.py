"""Precision/recall for detecting *any* plant, ignoring crop vs weed labels.

The main evaluation (evaluate_counts.py) reports crop and weed separately.
This script answers a different question: if you only care about "is there
a plant here at all," how well does the model do - merging both classes
into one "plant" category and matching predicted boxes to ground-truth
boxes by location (IoU), not by class label.

Usage:
    python scripts/evaluate_plant_detection.py --weights runs/colab_50epoch/best.pt
"""

import argparse
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

IMAGES_DIR = Path("data/yolo/images/val")
LABELS_DIR = Path("data/yolo/labels/val")
IOU_MATCH_THRESHOLD = 0.5


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def ground_truth_boxes(label_path: Path, width: int, height: int):
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            _, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
            boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def match(pred_boxes, gt_boxes):
    """Greedy IoU matching. pred_boxes must be sorted by confidence, descending."""
    matched_gt = set()
    tp = 0
    for pred in pred_boxes:
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= IOU_MATCH_THRESHOLD:
            matched_gt.add(best_idx)
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--conf", type=str, default="0.25", help="Comma-separated list of confidence thresholds to test, e.g. 0.25,0.35,0.45")
    parser.add_argument("--iou", type=float, default=0.3, help="NMS IoU threshold for the model's own predictions")
    args = parser.parse_args()

    conf_values = sorted(float(c) for c in args.conf.split(","))
    conf_floor = conf_values[0]

    model = YOLO(args.weights)
    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))

    totals = {c: {"tp": 0, "fp": 0, "fn": 0} for c in conf_values}

    for image_path in image_paths:
        label_path = LABELS_DIR / (image_path.stem + ".txt")
        width, height = Image.open(image_path).size
        gt_boxes = ground_truth_boxes(label_path, width, height)

        result = model.predict(source=str(image_path), conf=conf_floor, iou=args.iou, verbose=False)[0]
        # (box, confidence) pairs, sorted by confidence descending
        all_preds = sorted(
            ((tuple(box.xyxy[0].tolist()), box.conf.item()) for box in result.boxes),
            key=lambda p: -p[1],
        )

        for c in conf_values:
            pred_boxes = [box for box, conf in all_preds if conf >= c]
            tp, fp, fn = match(pred_boxes, gt_boxes)
            totals[c]["tp"] += tp
            totals[c]["fp"] += fp
            totals[c]["fn"] += fn

    print(f"Evaluated {len(image_paths)} validation images (any plant, class-agnostic, IoU>={IOU_MATCH_THRESHOLD})\n")
    print(f"{'conf':>6}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'precision':>9}  {'recall':>7}")
    for c in conf_values:
        tp, fp, fn = totals[c]["tp"], totals[c]["fp"], totals[c]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"{c:>6}  {tp:>5}  {fp:>5}  {fn:>5}  {precision:>9.3f}  {recall:>7.3f}")


if __name__ == "__main__":
    main()
