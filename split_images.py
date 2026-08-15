import cv2
import os
from pathlib import Path

INPUT_DIR = Path.home() / "Desktop" / "10_08_v_grass"
OUTPUT_DIR = Path.home() / "Desktop" / "10_08_v_grass_split"
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def split_image_into_9(image_path: Path, output_dir: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load: {image_path}")
        return

    height, width = image.shape[:2]
    h_step = height // 3
    w_step = width // 3

    stem = image_path.stem
    count = 1
    for row in range(3):
        for col in range(3):
            y1 = row * h_step
            y2 = (row + 1) * h_step if row < 2 else height
            x1 = col * w_step
            x2 = (col + 1) * w_step if col < 2 else width

            crop = image[y1:y2, x1:x2]
            out_path = output_dir / f"{stem}_part{count}.jpg"
            cv2.imwrite(str(out_path), crop)
            count += 1

    print(f"{image_path.name} -> split into 9 parts")

def main():
    image_files = [
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    print(f"Processing {len(image_files)} images")

    for image_path in image_files:
        split_image_into_9(image_path, OUTPUT_DIR)

    print(f"Done! Split images saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()