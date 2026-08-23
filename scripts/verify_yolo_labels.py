"""Sanity check: draw converted YOLO boxes back onto a few sample images.

Picks a handful of random images from data/yolo/images/train, draws their
label boxes on top (green = crop, red = weed), and saves them to
data/yolo/preview/ so we can eyeball whether the COCO -> YOLO conversion
produced correctly placed boxes before spending time training on it.
"""

import random
from pathlib import Path

import cv2

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "yolo"
IMAGES_DIR = DATA_DIR / "images" / "train"
LABELS_DIR = DATA_DIR / "labels" / "train"
PREVIEW_DIR = DATA_DIR / "preview"

CLASS_NAMES = ["crop", "weed"]
CLASS_COLORS = {0: (0, 200, 0), 1: (0, 0, 255)}  # BGR: green=crop, red=weed

N_SAMPLES = 5


def draw_boxes(image_path: Path, label_path: Path, out_path: Path) -> None:
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]

    lines = label_path.read_text().splitlines() if label_path.exists() else []
    for line in lines:
        class_id, cx, cy, w, h = line.split()
        class_id = int(class_id)
        cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        color = CLASS_COLORS[class_id]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

    n_crop = sum(1 for l in lines if l.startswith("0 "))
    n_weed = sum(1 for l in lines if l.startswith("1 "))
    label = f"crop={n_crop} weed={n_weed}"
    cv2.putText(image, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 3)

    cv2.imwrite(str(out_path), image)
    print(f"{image_path.name}: {label} -> {out_path}")


def main() -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    all_images = sorted(IMAGES_DIR.glob("*.jpg"))
    sample = random.sample(all_images, min(N_SAMPLES, len(all_images)))

    for image_path in sample:
        label_path = LABELS_DIR / (image_path.stem + ".txt")
        out_path = PREVIEW_DIR / image_path.name
        draw_boxes(image_path, label_path, out_path)

    print(f"\nOpen {PREVIEW_DIR} in File Explorer or VS Code to inspect the boxes.")


if __name__ == "__main__":
    main()
