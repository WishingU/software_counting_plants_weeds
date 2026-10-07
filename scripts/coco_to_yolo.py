"""Convert the COCO-format plant/weed dataset into YOLO format for counting.

Reads the 5 date-based *_combined_train.json / *_combined_val.json files
(the clean, non-overlapping canonical split) and writes a YOLO-style dataset:
one image copy + one label .txt per image, with the 4 original species
collapsed into 2 counting classes - 'crop' (wheat) and 'weed' (wild oat,
brome grass, barley grass) - since this stage counts plants and weeds
separately but doesn't yet identify weed species (that's step 2).

Source data is expected at ~/Downloads/training_data/train (not tracked in
git - see .gitignore). Output goes to data/yolo/ inside this repo, also not
tracked in git except for the small data.yaml config.
"""

import json
import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "yolo"

DATES = ["20230523", "20230526", "20230530", "20230602", "20230720"]
SPLITS = ["train", "val"]

# Original COCO category name -> merged counting class id/name
CLASS_NAMES = ["crop", "weed"]
CATEGORY_NAME_TO_CLASS_ID = {
    "wheat": 0,
    "wild oat": 1,
    "brome grass": 1,
    "barley grass": 1,
}


def convert_split(
    date: str,
    split: str,
    source_dir: Path,
    extra_source_dir: Path | None,
    output_dir: Path,
) -> None:
    json_path = source_dir / f"{date}_combined_{split}.json"
    coco = json.loads(json_path.read_text())

    category_id_to_class_id = {
        cat["id"]: CATEGORY_NAME_TO_CLASS_ID[cat["name"]] for cat in coco["categories"]
    }

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    img_out_dir = output_dir / "images" / split
    lbl_out_dir = output_dir / "labels" / split
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    n_images, n_missing = 0, 0
    n_boxes_by_class = {0: 0, 1: 0}
    for image_id, img in images_by_id.items():
        src_img_path = source_dir / img["file_name"]
        if not src_img_path.exists() and extra_source_dir is not None:
            src_img_path = extra_source_dir / img["file_name"]
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

    print(
        f"{date} [{split}]: {n_images} images, "
        f"{n_boxes_by_class[0]} crop boxes, {n_boxes_by_class[1]} weed boxes, "
        f"{n_missing} missing image files"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--extra-source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    extra_source_dir = args.extra_source_dir.resolve() if args.extra_source_dir else None
    output_dir = args.output_dir.resolve()
    for date in DATES:
        for split in SPLITS:
            convert_split(date, split, source_dir, extra_source_dir, output_dir)

    data_yaml = output_dir / "data.yaml"
    names_lines = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    data_yaml.write_text(
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names_lines}\n"
    )
    print(f"\nWrote dataset config to {data_yaml}")


if __name__ == "__main__":
    main()
