"""Convert the COCO-format dataset into YOLO format for species identification.

Same source data and conversion logic as coco_to_yolo.py, but keeps the 4
original species as separate classes instead of merging the 3 weed species
into one 'weed' class. This is for Item 2 (species identification), a
separate model from the Item 1 crop/weed counter.

Writes to data/yolo_species/ (not data/yolo/) so this doesn't disturb the
already-validated crop/weed counting dataset and model.
"""

import json
import shutil
from pathlib import Path

SOURCE_DIR = Path.home() / "Downloads" / "training_data" / "train"
EXTRA_SOURCE_DIR = Path.home() / "Downloads" / "training_data" / "train 2"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "yolo_species"

DATES = ["20230523", "20230526", "20230530", "20230602", "20230720"]
SPLITS = ["train", "val"]

# Original COCO category name -> class id/name (kept separate, not merged)
CLASS_NAMES = ["wheat", "wild oat", "brome grass", "barley grass"]
CATEGORY_NAME_TO_CLASS_ID = {
    "wheat": 0,
    "wild oat": 1,
    "brome grass": 2,
    "barley grass": 3,
}


def convert_split(date: str, split: str) -> None:
    json_path = SOURCE_DIR / f"{date}_combined_{split}.json"
    coco = json.loads(json_path.read_text())

    category_id_to_class_id = {
        cat["id"]: CATEGORY_NAME_TO_CLASS_ID[cat["name"]] for cat in coco["categories"]
    }

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    img_out_dir = OUTPUT_DIR / "images" / split
    lbl_out_dir = OUTPUT_DIR / "labels" / split
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    n_images, n_missing = 0, 0
    n_boxes_by_class = {0: 0, 1: 0, 2: 0, 3: 0}
    for image_id, img in images_by_id.items():
        src_img_path = SOURCE_DIR / img["file_name"]
        if not src_img_path.exists():
            src_img_path = EXTRA_SOURCE_DIR / img["file_name"]
        if not src_img_path.exists():
            n_missing += 1
            continue

        dst_img_path = img_out_dir / img["file_name"]
        if not dst_img_path.exists():
            shutil.copy2(src_img_path, dst_img_path)

        width, height = img["width"], img["height"]
        lines = []
        for ann in anns_by_image.get(image_id, []):
            class_id = category_id_to_class_id[ann["category_id"]]
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / width
            cy = (y + h / 2) / height
            nw = w / width
            nh = h / height
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_boxes_by_class[class_id] += 1

        label_path = lbl_out_dir / (Path(img["file_name"]).stem + ".txt")
        label_path.write_text("\n".join(lines))
        n_images += 1

    counts_str = ", ".join(f"{CLASS_NAMES[i]}={n_boxes_by_class[i]}" for i in range(4))
    print(f"{date} [{split}]: {n_images} images, {counts_str}, {n_missing} missing image files")


def main() -> None:
    for date in DATES:
        for split in SPLITS:
            convert_split(date, split)

    data_yaml = OUTPUT_DIR / "data.yaml"
    names_lines = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    data_yaml.write_text(
        f"path: {OUTPUT_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names_lines}\n"
    )
    print(f"\nWrote dataset config to {data_yaml}")


if __name__ == "__main__":
    main()
