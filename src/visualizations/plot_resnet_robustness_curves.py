"""Plot level-wise robustness curves from existing ResNet-18 robustness CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ResNet-18 robustness curves")
    parser.add_argument("--resnet_robustness_csv", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    return parser.parse_args()


def plot_robustness_curves(resnet_robustness_csv: Path, output_path: Path) -> Path:
    """Plot per-corruption accuracy versus corruption level."""
    df = pd.read_csv(resnet_robustness_csv)
    clean_row = df[df["corruption"] == "clean"].iloc[0]
    clean_acc = float(clean_row["acc"])

    mappings = [
        ("jpeg", "JPEG Compression", [75, 50, 25]),
        ("gaussian_blur", "Gaussian Blur", [1, 2, 3]),
        ("gaussian_noise", "Gaussian Noise", [0.05, 0.10, 0.20]),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, (name, title, level_order) in zip(axes, mappings):
        sub = df[df["corruption"] == name].copy()
        sub["level"] = pd.to_numeric(sub["level"], errors="coerce")
        sub = sub.set_index("level").reindex(level_order).reset_index()
        ax.plot(sub["level"], sub["acc"], marker="o", linewidth=2)
        ax.axhline(clean_acc, linestyle="--", color="gray", label="Clean Acc")
        ax.set_title(title)
        ax.set_xlabel("Corruption Level")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Accuracy")
    axes[2].legend(loc="best")
    fig.suptitle("ResNet-18 Robustness Curves", fontweight="bold")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    saved_path = plot_robustness_curves(args.resnet_robustness_csv, args.output_path)
    print(f"Saved: {saved_path}")


if __name__ == "__main__":
    main()
