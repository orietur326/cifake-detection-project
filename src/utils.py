"""General utility helpers shared by Part 2 modules."""

from __future__ import annotations

import random
import os
from pathlib import Path
from typing import Optional

import numpy as np


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def list_image_files(folder: Path, max_files: Optional[int] = None) -> list[Path]:
    """List image files under a folder with optional early stopping.

    When ``max_files`` is provided, scanning stops as soon as that many
    image paths are collected.
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    print(f"Entering folder scan: {folder}")

    image_paths: list[Path] = []
    with os.scandir(folder) as entries:
        for entry in entries:
            if max_files is not None and len(image_paths) >= max_files:
                break

            if not entry.is_file(follow_symlinks=False):
                continue

            suffix = Path(entry.name).suffix.lower()
            if suffix in exts:
                image_paths.append(folder / entry.name)

    if max_files is None:
        image_paths.sort()

    return image_paths
