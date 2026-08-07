"""Evaluate detection metrics and per-image counting error."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from plant_counter.metrics import summarize_counts


def _load_truth(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    if not path.exists():
        raise FileNotFoundError(f"Missing ground-truth label: {path}")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Malformed YOLO label {path}:{line_number}")
        counts[int(fields[0])] += 1
    return counts


def _resolve_data(data_path: Path, split: str) -> tuple[Path, Path, list[str]]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required; install the project dependencies first.") from exc
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(data["path"])
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    images_dir = (root / data[split]).resolve()
    relative_images = Path(data[split])
    if "images" not in relative_images.parts:
        raise ValueError("Dataset split path must contain an 'images' directory")
    parts = list(relative_images.parts)
    parts[parts.index("images")] = "labels"
    labels_dir = (root / Path(*parts)).resolve()
    names_data: Any = data["names"]
    if isinstance(names_data, dict):
        names = [str(names_data[index]) for index in sorted(names_data)]
    else:
        names = [str(name) for name in names_data]
    return images_dir, labels_dir, names


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate detection and counting performance.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed; run `python -m pip install -e .`.") from exc

    images_dir, labels_dir, class_names = _resolve_data(args.data.resolve(), args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model.resolve()))
    validation = model.val(
        data=str(args.data.resolve()),
        split=args.split,
        device=args.device,
        plots=True,
        workers=args.workers,
        project=str(args.output_dir.resolve()),
        name="detection",
        exist_ok=True,
    )
    results = model.predict(
        source=str(images_dir), conf=args.conf, iou=args.iou, device=args.device, stream=True
    )

    truth_counts: list[Counter[int]] = []
    predicted_counts: list[Counter[int]] = []
    rows: list[dict[str, object]] = []
    for result in results:
        image_path = Path(result.path)
        truth = _load_truth(labels_dir / f"{image_path.stem}.txt")
        predicted: Counter[int] = Counter()
        if result.boxes is not None:
            predicted.update(int(value) for value in result.boxes.cls.cpu().tolist())
        truth_counts.append(truth)
        predicted_counts.append(predicted)
        row: dict[str, object] = {
            "image": image_path.name,
            "truth_total": sum(truth.values()),
            "predicted_total": sum(predicted.values()),
        }
        for index, name in enumerate(class_names):
            row[f"truth_{name}"] = truth[index]
            row[f"predicted_{name}"] = predicted[index]
        rows.append(row)

    summary = summarize_counts(truth_counts, predicted_counts, class_names)
    summary["detection"] = {
        "precision": float(validation.box.mp),
        "recall": float(validation.box.mr),
        "map50": float(validation.box.map50),
        "map50_95": float(validation.box.map),
    }
    summary["settings"] = {"confidence": args.conf, "iou": args.iou, "split": args.split}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if rows:
        with (args.output_dir / "per_image_counts.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
