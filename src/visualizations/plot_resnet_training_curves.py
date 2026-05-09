"""Plot ResNet-18 training curves from an existing training log CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REQUIRED_COLUMNS = ["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training curve plotting."""
    parser = argparse.ArgumentParser(description="Plot ResNet-18 training curves from training_log.csv")
    parser.add_argument("--training_log_csv", type=Path, required=True, help="Path to training_log.csv")
    parser.add_argument("--output_path", type=Path, required=True, help="Path for output PNG")
    return parser.parse_args()


def plot_training_curves(training_log_csv: Path, output_path: Path) -> Path:
    """Create train/validation loss and accuracy curves and save as PNG."""
    df = pd.read_csv(training_log_csv)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in training log: {missing}")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(df["epoch"], df["val_loss"], marker="o", label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_accuracy"], marker="o", label="Train Accuracy")
    axes[1].plot(df["epoch"], df["val_accuracy"], marker="o", label="Val Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.suptitle("ResNet-18 Training Curves", fontweight="bold")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    saved_path = plot_training_curves(args.training_log_csv, args.output_path)
    print(f"Saved: {saved_path}")


if __name__ == "__main__":
    main()
