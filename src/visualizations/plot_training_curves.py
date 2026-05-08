"""Plot CIFAKE ResNet-18 training curves from training_log.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_COLUMNS = ["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for plotting training curves."""
    parser = argparse.ArgumentParser(description="Plot train/validation curves for CIFAKE ResNet-18")
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("outputs/resnet18/training_log.csv"),
        help="Path to training log CSV",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("outputs/figures/training_curves.png"),
        help="Path to save training curves plot",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI for report-ready output")
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    """Validate required training log columns exist."""
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in training log CSV: {missing}")


def main() -> None:
    """Entry point for plotting training curves."""
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Training log CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    validate_columns(df)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    axes[0, 0].plot(df["epoch"], df["train_loss"], color="#1f77b4", marker="o", linewidth=1.8)
    axes[0, 0].set_title("Train Loss vs Epoch", fontweight="bold")
    axes[0, 0].set_ylabel("Loss")

    axes[0, 1].plot(df["epoch"], df["val_loss"], color="#d62728", marker="o", linewidth=1.8)
    axes[0, 1].set_title("Validation Loss vs Epoch", fontweight="bold")

    axes[1, 0].plot(df["epoch"], df["train_accuracy"], color="#2ca02c", marker="o", linewidth=1.8)
    axes[1, 0].set_title("Train Accuracy vs Epoch", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")

    axes[1, 1].plot(df["epoch"], df["val_accuracy"], color="#9467bd", marker="o", linewidth=1.8)
    axes[1, 1].set_title("Validation Accuracy vs Epoch", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")

    for ax in axes.flat:
        ax.grid(alpha=0.35)

    fig.suptitle("ResNet-18 Training Curves on CIFAKE", fontsize=14, fontweight="bold")
    fig.tight_layout()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves figure: {args.output_path}")


if __name__ == "__main__":
    main()
