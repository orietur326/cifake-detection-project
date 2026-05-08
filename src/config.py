"""Configuration constants for CIFAKE Part 2 project foundation."""

from pathlib import Path

# Label mapping required by project specification.
LABEL_TO_INDEX = {"REAL": 0, "FAKE": 1}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}

# Expected CIFAKE directory names.
TRAIN_SPLIT_NAME = "train"
TEST_SPLIT_NAME = "test"

# Robustness settings (must match Part 1 exactly).
JPEG_QUALITIES = (75, 50, 25)
GAUSSIAN_BLUR_SIGMAS = (1.0, 2.0, 3.0)
GAUSSIAN_NOISE_SIGMAS = (0.05, 0.10, 0.20)

# Default random seed requested in AGENTS instructions.
DEFAULT_SEED = 42


def expected_cifake_structure(data_dir: Path) -> dict[str, Path]:
    """Return expected CIFAKE class directory paths for quick validation messages."""
    return {
        "train_real": data_dir / TRAIN_SPLIT_NAME / "REAL",
        "train_fake": data_dir / TRAIN_SPLIT_NAME / "FAKE",
        "test_real": data_dir / TEST_SPLIT_NAME / "REAL",
        "test_fake": data_dir / TEST_SPLIT_NAME / "FAKE",
    }
