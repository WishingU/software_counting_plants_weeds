"""Audit detections that exist at conf=0.25 but get filtered out at conf=0.35.

Specifically targets unmatched detections (would-be "false positives") in the
confidence band [0.25, 0.35) that don't match any ground-truth label - these
are the ones raising the threshold silently discards. If most of them turn
out to be real plants (same pattern as audit_false_positives.py found),
that confirms the threshold change is trading away real detections, not
just filtering noise.

Usage:
    python scripts/audit_threshold_gap.py --weights runs/colab_50epoch/best.pt --n 30
"""

import argparse
import random
from pathlib import Path

import cv2
from PIL import Image, ImageDraw
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "yolo" / "images" / "val"
LABELS_DIR = PROJECT_ROOT / "data" / "yolo" / "labels" / "val"
OUT_DIR = PROJECT_ROOT / "outputs" / "threshold_gap_audit"
IOU_MATCH_THRESHOLD = 0.5
CELL_SIZE = 340
GRID_COLS = 5
GRID_ROWS = 2
PAD = 40


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
    parser.add_argument("--low", type=float, default=0.25)
    parser.add_argument("--high", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(args.weights)
    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))

    gap_unmatched = []  # unmatched detections in [low, high) confidence band

    for image_path in image_paths:
        label_path = LABELS_DIR / (image_path.stem + ".txt")
        width, height = Image.open(image_path).size
        gt_boxes = ground_truth_boxes(label_path, width, height)

        result = model.predict(source=str(image_path), conf=args.low, iou=args.iou, device=args.device, verbose=False)[0]
        preds = [(tuple(box.xyxy[0].tolist()), box.conf.item()) for box in result.boxes]
        preds.sort(key=lambda p: -p[1])

        matched_gt = set()
        for box, conf in preds:
            best_iou, best_idx = 0.0, -1
            for i, gt in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                iou = box_iou(box, gt)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            is_match = best_iou >= IOU_MATCH_THRESHOLD
            if is_match:
                matched_gt.add(best_idx)
            elif args.low <= conf < args.high:
                gap_unmatched.append((image_path, box, width, height, conf))

    print(f"Unmatched detections in confidence band [{args.low}, {args.high}): {len(gap_unmatched)}")

    random.seed(args.seed)
    sample = random.sample(gap_unmatched, min(args.n, len(gap_unmatched)))
    print(f"Randomly sampled {len(sample)} for audit (seed={args.seed})\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_grid = GRID_COLS * GRID_ROWS
    n_grids = (len(sample) + per_grid - 1) // per_grid

    for grid_idx in range(n_grids):
        grid_samples = sample[grid_idx * per_grid: (grid_idx + 1) * per_grid]
        sheet = Image.new("RGB", (CELL_SIZE * GRID_COLS, CELL_SIZE * GRID_ROWS), (30, 30, 30))
        draw = ImageDraw.Draw(sheet)

        for i, (image_path, box, width, height, conf) in enumerate(grid_samples):
            global_idx = grid_idx * per_grid + i
            x1, y1, x2, y2 = box
            cx1 = max(0, int(x1) - PAD)
            cy1 = max(0, int(y1) - PAD)
            cx2 = min(width, int(x2) + PAD)
            cy2 = min(height, int(y2) + PAD)

            cv_img = cv2.imread(str(image_path))
            cv2.rectangle(cv_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            crop = cv_img[cy1:cy2, cx1:cx2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_img = Image.fromarray(crop_rgb).resize((CELL_SIZE - 10, CELL_SIZE - 30))

            row, col = divmod(i, GRID_COLS)
            paste_x, paste_y = col * CELL_SIZE + 5, row * CELL_SIZE + 25
            sheet.paste(crop_img, (paste_x, paste_y))
            draw.text((col * CELL_SIZE + 8, row * CELL_SIZE + 4), f"#{global_idx} ({conf:.2f})", fill=(255, 255, 0))

        out_path = OUT_DIR / f"gap_grid_{grid_idx}.png"
        sheet.save(out_path)
        print(f"Saved {out_path}")

    print("\nEach cell shows a detection that exists at conf=0.25 but is filtered out at conf=0.35.")


if __name__ == "__main__":
    main()
