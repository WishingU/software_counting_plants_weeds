"""Train a YOLOv8 model to detect crop/weed plants, or identify species.

Usage:
    python scripts/train.py --epochs 3                                          # quick smoke test, crop/weed
    python scripts/train.py --epochs 50                                         # real training run, crop/weed
    python scripts/train.py --epochs 50 --data data/yolo_species/data.yaml --name species_id  # species model

Uses Apple Silicon's MPS backend if available, falling back to CPU.
Results (weights, metrics, sample predictions) are saved under runs/,
which is gitignored - trained weights are large binaries that don't
belong in git.
"""

import argparse

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/yolo/data.yaml", help="Path to dataset yaml")
    parser.add_argument("--name", type=str, default="crop_weed_counter", help="Run name under runs/")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on device: {device}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=device,
        project="runs",
        name=args.name,
    )


if __name__ == "__main__":
    main()
