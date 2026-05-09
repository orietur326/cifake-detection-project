"""Create corruption examples for one REAL and one FAKE CIFAKE image."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from src.corruptions import apply_gaussian_blur, apply_gaussian_noise, apply_jpeg_compression
from src.utils import list_image_files, set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI args for corruption-example plotting."""
    parser = argparse.ArgumentParser(description="Generate REAL/FAKE corruption example figure")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to CIFAKE root directory")
    parser.add_argument("--output_path", type=Path, required=True, help="Path for output PNG")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_first_image(data_dir: Path, class_name: str) -> Image.Image:
    class_dir = data_dir / "test" / class_name
    paths = list_image_files(class_dir)
    if not paths:
        raise FileNotFoundError(f"No images found in {class_dir}")
    return Image.open(paths[0]).convert("RGB")


def generate_corruption_examples(data_dir: Path, output_path: Path, seed: int = 42) -> Path:
    """Generate corruption_examples.png with required corruption settings."""
    set_seed(seed)
    real = _load_first_image(data_dir, "REAL")
    fake = _load_first_image(data_dir, "FAKE")

    variants = [
        ("Original", lambda img: img),
        ("JPEG-25", lambda img: apply_jpeg_compression(img, 25)),
        ("Gaussian Blur σ=3", lambda img: apply_gaussian_blur(img, 3)),
        ("Gaussian Noise σ=0.2", lambda img: apply_gaussian_noise(img, 0.20)),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, len(variants), figsize=(12, 6))
    rows = [("REAL (0)", real), ("FAKE (1)", fake)]

    for r, (row_name, source) in enumerate(rows):
        for c, (title, transform) in enumerate(variants):
            ax = axes[r, c]
            ax.imshow(transform(source))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(row_name, fontweight="bold")

    fig.suptitle("CIFAKE Corruption Examples", fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    output_path = generate_corruption_examples(args.data_dir, args.output_path, args.seed)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
