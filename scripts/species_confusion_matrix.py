"""Generate a confusion matrix for the species identification model.

Runs a validation pass (fast - no training) on the species model and saves
the confusion matrix plot, so we can see specifically which species get
mixed up with which, rather than just an aggregate accuracy number.

Usage:
    python scripts/species_confusion_matrix.py --weights runs/colab_species_50epoch/best.pt
"""

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="data/yolo_species/data.yaml")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, plots=True, project="runs", name="species_confusion")

    print(f"\nResults and confusion matrix saved to: {metrics.save_dir}")
    print("Look for confusion_matrix.png and confusion_matrix_normalized.png in that folder.")


if __name__ == "__main__":
    main()
