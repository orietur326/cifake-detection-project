"""Data loading utilities for CIFAKE with deterministic subset controls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from sklearn.model_selection import train_test_split

from .config import LABEL_TO_INDEX, TEST_SPLIT_NAME, TRAIN_SPLIT_NAME, expected_cifake_structure
from .utils import list_image_files, set_seed


@dataclass(frozen=True)
class Sample:
    """A single CIFAKE sample reference with label index."""

    image_path: Path
    label: int


def _sample_paths(paths: list[Path], max_count: Optional[int], seed: int) -> list[Path]:
    if max_count is None or max_count >= len(paths):
        return paths
    import random

    rng = random.Random(seed)
    chosen = rng.sample(paths, k=max_count)
    return sorted(chosen)


def _collect_split(data_dir: Path, split: str, max_each: Optional[int], seed: int) -> list[Sample]:
    samples: list[Sample] = []
    for class_name, label_idx in LABEL_TO_INDEX.items():
        class_dir = data_dir / split / class_name
        if not class_dir.exists():
            continue
        class_paths = list_image_files(class_dir)
        print(f"Collected {len(class_paths)} file paths for {split}/{class_name} before limiting.")
        class_paths = _sample_paths(class_paths, max_each, seed)
        if max_each is not None:
            print(f"Using {len(class_paths)} file paths for {split}/{class_name} after max_each={max_each}.")
        samples.extend(Sample(image_path=p, label=label_idx) for p in class_paths)
    return samples


def validate_cifake_dirs(data_dir: Path) -> None:
    """Validate expected CIFAKE directory layout and raise helpful errors if missing."""
    expected = expected_cifake_structure(data_dir)
    missing = [name for name, path in expected.items() if not path.exists()]
    if missing:
        details = "\n".join(f"- {name}: {expected[name]}" for name in missing)
        raise FileNotFoundError(
            f"Missing CIFAKE folders under data_dir={data_dir}. Expected:\n{details}"
        )
    print(f"Validated CIFAKE directory structure at: {data_dir}")


def load_cifake_samples(
    data_dir: Path,
    *,
    max_train_each: Optional[int] = None,
    max_test_each: Optional[int] = None,
    seed: int = 42,
    fast_dev_run: bool = False,
) -> tuple[list[Sample], list[Sample]]:
    """Load CIFAKE train/test sample references with deterministic optional caps.

    Returns lists of ``Sample`` entries for train and test splits.
    """
    data_dir = Path(data_dir)
    validate_cifake_dirs(data_dir)
    set_seed(seed)

    if fast_dev_run:
        max_train_each = min(max_train_each or 20, 20)
        max_test_each = min(max_test_each or 20, 20)
        print(
            "fast_dev_run enabled in data loading: "
            f"max_train_each={max_train_each}, max_test_each={max_test_each}"
        )

    train_samples = _collect_split(data_dir, TRAIN_SPLIT_NAME, max_train_each, seed)
    test_samples = _collect_split(data_dir, TEST_SPLIT_NAME, max_test_each, seed)
    print(
        f"Collected sample counts -> train: {len(train_samples)}, test: {len(test_samples)}"
    )
    return train_samples, test_samples


def split_train_val(
    train_samples: Iterable[Sample],
    *,
    val_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    """Create deterministic stratified train/validation split from train samples."""
    train_samples = list(train_samples)
    if not train_samples:
        return [], []

    y = [s.label for s in train_samples]
    train_idx, val_idx = train_test_split(
        range(len(train_samples)),
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    train_split = [train_samples[i] for i in train_idx]
    val_split = [train_samples[i] for i in val_idx]
    return train_split, val_split
