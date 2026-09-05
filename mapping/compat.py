"""Make the existing `experiments/common` package importable without moving it.

Usage:
    from mapping import compat
    compat.ensure_experiments_on_path()
    from common import io_data, class_map, camera
"""
from __future__ import annotations

import sys

from .config import EXPERIMENTS_DIR


def ensure_experiments_on_path() -> None:
    p = str(EXPERIMENTS_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
