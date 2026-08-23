"""Visualize false positives to understand what's causing them.

Finds the validation images with the most false-positive detections (boxes
the model drew that don't match any real plant), and saves annotated copies
color-coded so the failure pattern is visible at a glance:
  green  = true positive  (correct detection)
  red    = false positive (model hallucinated a plant)
  yellow = false negative (real plant the model missed)

Usage:
    python scripts/diagnose_false_positives.py --weights runs/colab_50epoch/best.pt
"""

import argparse
from pathlib import Path

import cv2
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "yolo" / "images" / "val"
LABELS_DIR = PROJECT_ROOT / "data" / "yolo" / "labels" / "val"
OUT_DIR = PROJECT_ROOT / "outputs" / "diagnose_fp"
IOU_MATCH_THRESHOLD = 0.5
N_WORST = 5


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(args.weights)
    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))

    per_image = []
    for image_path in image_paths:
        label_path = LABELS_DIR / (image_path.stem + ".txt")
        width, height = Image.open(image_path).size
        gt_boxes = ground_truth_boxes(label_path, width, height)

        result = model.predict(source=str(image_path), conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
        pred_boxes = [tuple(box.xyxy[0].tolist()) for box in result.boxes]

        matched_gt = set()
        matched_pred = set()
        for pi, pred in enumerate(pred_boxes):
            best_iou, best_idx = 0.0, -1
            for gi, gt in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = box_iou(pred, gt)
                if iou > best_iou:
                    best_iou, best_idx = iou, gi
            if best_iou >= IOU_MATCH_THRESHOLD:
                matched_gt.add(best_idx)
                matched_pred.add(pi)

        fp_boxes = [b for i, b in enumerate(pred_boxes) if i not in matched_pred]
        fn_boxes = [b for i, b in enumerate(gt_boxes) if i not in matched_gt]
        tp_boxes = [b for i, b in enumerate(pred_boxes) if i in matched_pred]

        per_image.append({
            "path": image_path, "tp": tp_boxes, "fp": fp_boxes, "fn": fn_boxes,
        })

    per_image.sort(key=lambda r: len(r["fp"]), reverse=True)
    worst = per_image[:N_WORST]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Top {N_WORST} images by false-positive count:\n")
    for r in worst:
        image = cv2.imread(str(r["path"]))
        for x1, y1, x2, y2 in r["tp"]:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 3)
        for x1, y1, x2, y2 in r["fp"]:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        for x1, y1, x2, y2 in r["fn"]:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 255), 3)

        out_path = OUT_DIR / r["path"].name
        cv2.imwrite(str(out_path), image)
        print(f"  {r['path'].name}: {len(r['fp'])} false positives, {len(r['fn'])} false negatives, {len(r['tp'])} correct -> {out_path}")

    print("\nGreen = correct, Red = false positive (hallucinated), Yellow = false negative (missed)")


if __name__ == "__main__":
    main()
