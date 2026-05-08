"""Generate corruption visualization examples for CIFAKE REAL and FAKE images."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from src.config import LABEL_TO_INDEX
from src.corruptions import apply_gaussian_blur, apply_gaussian_noise, apply_jpeg_compression
from src.utils import list_image_files, set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for corruption visualization generation."""
    parser = argparse.ArgumentParser(
        description="Create CIFAKE corruption examples figure for one REAL and one FAKE image"
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data/CIFAKE"), help="Path to CIFAKE root")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("outputs/figures/corruption_examples.png"),
        help="Path to save corruption visualization figure",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI for report-ready output")
    return parser.parse_args()


def load_one_image(data_dir: Path, class_name: str) -> Image.Image:
    """Load one image from CIFAKE train split for the given class name."""
    class_dir = data_dir / "train" / class_name
    paths = list_image_files(class_dir, max_files=None)
    if not paths:
        raise FileNotFoundError(f"No images found for class '{class_name}' in {class_dir}")
    return Image.open(paths[0]).convert("RGB")


def main() -> None:
    """Entry point for generating CIFAKE corruption examples visualization."""
    args = parse_args()
    set_seed(args.seed)

    real_image = load_one_image(args.data_dir, "REAL")
    fake_image = load_one_image(args.data_dir, "FAKE")

    corruption_levels = {
        "Original": None,
        "JPEG q=75": ("jpeg", 75),
        "JPEG q=50": ("jpeg", 50),
        "JPEG q=25": ("jpeg", 25),
        "Blur σ=1": ("blur", 1),
        "Blur σ=2": ("blur", 2),
        "Blur σ=3": ("blur", 3),
        "Noise σ=0.05": ("noise", 0.05),
        "Noise σ=0.10": ("noise", 0.10),
        "Noise σ=0.20": ("noise", 0.20),
    }

    fig, axes = plt.subplots(2, len(corruption_levels), figsize=(24, 5))
    plt.style.use("seaborn-v0_8-whitegrid")

    row_meta = [("REAL (Label 0)", real_image), ("FAKE (Label 1)", fake_image)]

    for row_idx, (row_label, source_image) in enumerate(row_meta):
        for col_idx, (title, setting) in enumerate(corruption_levels.items()):
            ax = axes[row_idx, col_idx]
            if setting is None:
                rendered = source_image
            else:
                kind, level = setting
                if kind == "jpeg":
                    rendered = apply_jpeg_compression(source_image, int(level))
                elif kind == "blur":
                    rendered = apply_gaussian_blur(source_image, float(level))
                else:
                    rendered = apply_gaussian_noise(source_image, float(level))

            ax.imshow(rendered)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=11, fontweight="bold")

    fig.suptitle(
        "CIFAKE Corruption Examples: REAL vs FAKE Under JPEG, Blur, and Noise",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved corruption visualization: {args.output_path}")
    print(f"Label mapping used: {LABEL_TO_INDEX}")


if __name__ == "__main__":
    main()
