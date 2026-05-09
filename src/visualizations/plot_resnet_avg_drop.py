"""Plot average corruption accuracy drop as bar chart and heatmap."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CORRUPTION_LABELS = {
    "jpeg": "JPEG",
    "gaussian_blur": "Gaussian Blur",
    "gaussian_noise": "Gaussian Noise",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot average accuracy drop for ResNet-18 robustness")
    parser.add_argument("--resnet_robustness_csv", type=Path, required=True)
    parser.add_argument("--bar_output_path", type=Path, required=True)
    parser.add_argument("--heatmap_output_path", type=Path, required=True)
    return parser.parse_args()


def compute_avg_drop(resnet_robustness_csv: Path) -> pd.DataFrame:
    """Compute average clean-to-corrupted accuracy drop by corruption type."""
    df = pd.read_csv(resnet_robustness_csv)
    sub = df[df["corruption"] != "clean"].copy()
    avg = sub.groupby("corruption", as_index=False)["drop"].mean()
    avg["corruption_label"] = avg["corruption"].map(CORRUPTION_LABELS)
    return avg


def plot_avg_drop(avg_df: pd.DataFrame, bar_output_path: Path, heatmap_output_path: Path) -> None:
    """Save bar and heatmap figures for average drops."""
    ordered = avg_df.sort_values("drop", ascending=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(ordered["corruption_label"], ordered["drop"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Average Accuracy Drop by Corruption Type")
    ax.set_ylabel("Average Drop (Clean Acc - Corrupted Acc)")
    ax.set_xlabel("Corruption Type")
    fig.tight_layout()
    bar_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(bar_output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 2.5))
    heat_df = ordered[["corruption_label", "drop"]].set_index("corruption_label").T
    im = ax.imshow(heat_df.values, aspect="auto", cmap="Reds")
    ax.set_xticks(range(len(heat_df.columns)))
    ax.set_xticklabels(heat_df.columns)
    ax.set_yticks([0])
    ax.set_yticklabels(["Avg Drop"])
    for i, col in enumerate(heat_df.columns):
        ax.text(i, 0, f"{heat_df.iloc[0, i]:.4f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Average Drop Heatmap")
    fig.tight_layout()
    fig.savefig(heatmap_output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    avg_df = compute_avg_drop(args.resnet_robustness_csv)
    plot_avg_drop(avg_df, args.bar_output_path, args.heatmap_output_path)
    print(f"Saved: {args.bar_output_path}")
    print(f"Saved: {args.heatmap_output_path}")


if __name__ == "__main__":
    main()
