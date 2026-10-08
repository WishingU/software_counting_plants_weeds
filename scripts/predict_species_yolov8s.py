"""Run species detection and counting using the trained YOLOv8s model."""

from pathlib import Path
import argparse

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path("models") / "species" / "yolov8s_full.pt"


def main():
    parser = argparse.ArgumentParser(
        description="Detect and count plant species using YOLOv8s."
    )

    parser.add_argument(
        "source",
        help="Path to an image, folder, or video.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )

    args = parser.parse_args()

    full_model_path = BASE_DIR / MODEL_PATH

    if not full_model_path.is_file():
        raise FileNotFoundError(
            f"YOLOv8s model not found: {full_model_path}"
        )

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=640,
        save=True,
        project=str(BASE_DIR / "runs"),
        name="species_yolov8s_predict",
        exist_ok=True,
    )

    names = {
        int(class_id): str(name)
        for class_id, name in model.names.items()
    }

    for result in results:
        counts = {name: 0 for name in names.values()}

        if result.boxes is not None:
            for class_id in result.boxes.cls.int().cpu().tolist():
                species = names[class_id]
                counts[species] += 1

        print(f"\nImage: {Path(result.path).name}")
        print(f"Total plants: {sum(counts.values())}")

        for species, count in counts.items():
            print(f"{species}: {count}")


if __name__ == "__main__":
    main()