"""Counting metrics independent of the model runtime."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def summarize_counts(
    truth: Sequence[Mapping[int, int]],
    predictions: Sequence[Mapping[int, int]],
    class_names: Sequence[str],
) -> dict[str, object]:
    if len(truth) != len(predictions):
        raise ValueError("Truth and prediction collections must have equal length")
    if not truth:
        raise ValueError("At least one image is required")

    total_errors: list[int] = []
    per_class_errors: dict[int, list[int]] = {index: [] for index in range(len(class_names))}
    for truth_counts, predicted_counts in zip(truth, predictions):
        truth_total = sum(int(truth_counts.get(index, 0)) for index in per_class_errors)
        predicted_total = sum(int(predicted_counts.get(index, 0)) for index in per_class_errors)
        total_errors.append(predicted_total - truth_total)
        for index in per_class_errors:
            per_class_errors[index].append(
                int(predicted_counts.get(index, 0)) - int(truth_counts.get(index, 0))
            )

    def metrics(errors: Sequence[int]) -> dict[str, float]:
        count = len(errors)
        return {
            "mae": sum(abs(error) for error in errors) / count,
            "rmse": math.sqrt(sum(error * error for error in errors) / count),
            "bias": sum(errors) / count,
            "exact_match_rate": sum(error == 0 for error in errors) / count,
        }

    return {
        "images": len(truth),
        "overall": metrics(total_errors),
        "per_class": {
            class_names[index]: metrics(errors) for index, errors in per_class_errors.items()
        },
    }
