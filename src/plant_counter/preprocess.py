"""Image preprocessing shared by interactive and command-line inference."""

from __future__ import annotations

from typing import Any

import numpy as np


def enhance_green(image: Any, strength: float = 0.4) -> np.ndarray:
    """Enhance pixels already dominated by green without tinting neutral areas.

    The returned image is a new RGB uint8 array. The adaptive mask makes the
    effect strongest where green exceeds the average red/blue level, so soil,
    labels, and grey backgrounds remain largely unchanged.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")

    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("image must be an RGB array with shape (height, width, 3)")
    if source.dtype != np.uint8:
        raise ValueError("image must use uint8 channel values")

    result = source.copy()
    channels = source.astype(np.float32)
    red, green, blue = channels[..., 0], channels[..., 1], channels[..., 2]
    green_dominance = np.clip(green - (red + blue) / 2.0, 0.0, 255.0) / 255.0
    boosted_green = green + strength * (255.0 - green) * green_dominance
    result[..., 1] = np.clip(np.rint(boosted_green), 0, 255).astype(np.uint8)
    return result
