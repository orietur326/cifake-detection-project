"""General utility helpers shared by Part 2 modules."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def list_image_files(folder: Path) -> list[Path]:
    """List common image files under a folder in sorted order."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])
