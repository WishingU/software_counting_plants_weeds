"""Train a YOLO model on Windows with automatic NVIDIA GPU selection.

Usage:
    python scripts/train.py --epochs 3     # quick smoke test
    python scripts/train.py --epochs 50    # real training run

Uses NVIDIA CUDA when available and falls back to CPU. Paths are resolved
from this file, so the command works from any current working directory.
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "data" / "yolo" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", type=str, default="yolo8m.pt")
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", help="CUDA device such as 0, or cpu (default: auto)")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--name", default="crop_weed_v1")
    args = parser.parse_args()

    data_path = args.data.resolve()
    if not data_path.is_file():
        raise SystemExit(f"Dataset configuration does not exist: {data_path}")
    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        seed=0,
        deterministic=True,
        project=str(RUNS_DIR),
        name=args.name,
    )


if __name__ == "__main__":
    main()
