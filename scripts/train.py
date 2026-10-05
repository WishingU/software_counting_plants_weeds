"""Compatibility entry point for the maintained training CLI.

This wrapper keeps ``python scripts/train.py`` working while all argument
definitions and reproducibility metadata remain in ``plant_counter.train``.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from plant_counter.train import main  # noqa: E402


if __name__ == "__main__":
    main()
