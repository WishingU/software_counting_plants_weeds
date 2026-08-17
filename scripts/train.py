"""Train a YOLOv8 model to detect crop and weed plants for counting.

Usage:
    python scripts/train.py --epochs 3     # quick smoke test
    python scripts/train.py --epochs 50    # real training run

Uses Apple Silicon's MPS backend if available, falling back to CPU.
Results (weights, metrics, sample predictions) are saved under runs/,
which is gitignored - trained weights are large binaries that don't
belong in git.
"""

import argparse

import torch
from ultralytics import YOLO

DATA_YAML = "data/yolo/data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on device: {device}")

    model = YOLO(args.model)
    model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=device,
        project="runs",
        name="crop_weed_counter",
    )


if __name__ == "__main__":
    main()
