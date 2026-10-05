"""Reproducible Ultralytics training entry point."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the detection baseline.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Integer batch size, or -1 for Ultralytics automatic batch selection.",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--lrf", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--close-mosaic", type=int)
    parser.add_argument("--warmup-epochs", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=Path, default=Path("outputs/runs/detect"))
    parser.add_argument("--name", default="baseline_v1")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Dataset YAML does not exist: {args.data}")
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Create a virtual environment and run "
            "`python -m pip install -e .` first."
        ) from exc

    args.project.mkdir(parents=True, exist_ok=True)
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    metadata_path = args.project / f"{args.name}_environment.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    model = YOLO(args.model)
    train_arguments = dict(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        project=str(args.project.resolve()),
        name=args.name,
    )
    for name in ("lr0", "lrf", "weight_decay", "close_mosaic", "warmup_epochs"):
        value = getattr(args, name)
        if value is not None:
            train_arguments[name] = value
    if args.optimizer != "auto":
        train_arguments["optimizer"] = args.optimizer
    model.train(**train_arguments)


if __name__ == "__main__":
    main()
