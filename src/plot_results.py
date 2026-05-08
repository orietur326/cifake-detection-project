"""Create final CIFAKE comparison tables and robustness plots.

This script combines:
- Fixed Part 1 ML clean metrics (Logistic Regression, LinearSVC, XGBoost)
- Optional Part 1 ML robustness CSV
- ResNet-18 clean and robustness CSVs

It outputs clean comparison CSV and, when possible, robustness plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ML_BASELINE_CLEAN = [
    {"model": "Logistic Regression", "accuracy": 0.8282, "f1": 0.8288},
    {"model": "LinearSVC", "accuracy": 0.8283, "f1": 0.8290},
    {"model": "XGBoost", "accuracy": 0.9071, "f1": 0.9082},
]

CORRUPTION_DISPLAY = {
    "jpeg": "JPEG compression",
    "gaussian_blur": "Gaussian blur",
    "gaussian_noise": "Gaussian noise",
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for final comparison generation."""
    parser = argparse.ArgumentParser(description="Build CIFAKE final comparison tables and plots")
    parser.add_argument(
        "--ml_robustness_csv",
        type=Path,
        default=Path("outputs/robustness_results.csv"),
        help="CSV for Part 1 ML robustness results",
    )
    parser.add_argument(
        "--resnet_clean_csv",
        type=Path,
        default=Path("outputs/resnet18/resnet18_clean_results.csv"),
        help="CSV for ResNet-18 clean metrics",
    )
    parser.add_argument(
        "--resnet_robustness_csv",
        type=Path,
        default=Path("outputs/resnet18/resnet18_robustness_results.csv"),
        help="CSV for ResNet-18 robustness results",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/final_comparison"),
        help="Directory for final comparison outputs",
    )
    return parser.parse_args()


def warn_missing(path: Path, label: str) -> None:
    """Print a consistent warning for missing optional input files."""
    print(f"[WARNING] {label} not found: {path}")


def read_csv_if_exists(path: Path, label: str) -> pd.DataFrame | None:
    """Read CSV if present; otherwise return None and warn."""
    if not path.exists():
        warn_missing(path, label)
        return None
    return pd.read_csv(path)


def normalize_robustness_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize robustness dataframe columns to expected schema."""
    out = df.copy()

    # common aliases for accuracy
    if "acc" not in out.columns and "accuracy" in out.columns:
        out["acc"] = out["accuracy"]
    if "clean_acc" not in out.columns and "clean_accuracy" in out.columns:
        out["clean_acc"] = out["clean_accuracy"]

    if "drop" not in out.columns and {"clean_acc", "acc"}.issubset(out.columns):
        out["drop"] = out["clean_acc"] - out["acc"]

    required = {"model", "corruption", "level", "acc", "clean_acc", "drop"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"Robustness CSV missing required columns: {sorted(missing)}")

    return out


def build_clean_comparison(resnet_clean_df: pd.DataFrame | None) -> pd.DataFrame:
    """Build clean metrics comparison for all available models."""
    rows = list(ML_BASELINE_CLEAN)

    if resnet_clean_df is not None:
        if {"accuracy", "f1"}.issubset(resnet_clean_df.columns):
            rows.append(
                {
                    "model": "ResNet-18",
                    "accuracy": float(resnet_clean_df.iloc[0]["accuracy"]),
                    "f1": float(resnet_clean_df.iloc[0]["f1"]),
                }
            )
        else:
            print("[WARNING] ResNet clean CSV found but does not contain 'accuracy' and 'f1'.")

    return pd.DataFrame(rows, columns=["model", "accuracy", "f1"])


def plot_robustness_curves(robust_df: pd.DataFrame, output_path: Path) -> None:
    """Plot all-model accuracy curves under each corruption type."""
    filtered = robust_df[robust_df["corruption"].isin(CORRUPTION_DISPLAY)].copy()
    if filtered.empty:
        print("[WARNING] No valid robustness rows for curve plotting.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

    for ax, corruption in zip(axes, ["jpeg", "gaussian_blur", "gaussian_noise"]):
        subset = filtered[filtered["corruption"] == corruption].copy()
        if subset.empty:
            ax.set_title(f"{CORRUPTION_DISPLAY[corruption]} (no data)")
            ax.set_xlabel("Corruption level")
            continue

        subset["level_numeric"] = pd.to_numeric(subset["level"], errors="coerce")
        subset = subset.sort_values(by="level_numeric", ascending=True)

        for model_name, group in subset.groupby("model"):
            ax.plot(group["level_numeric"], group["acc"], marker="o", label=model_name)

        ax.set_title(CORRUPTION_DISPLAY[corruption])
        ax.set_xlabel("Corruption level")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("CIFAKE Robustness Curves: Accuracy vs Corruption Level", y=1.03)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved robustness curves: {output_path}")


def plot_drop_heatmap(robust_df: pd.DataFrame, output_path: Path) -> None:
    """Plot heatmap of average accuracy drop by model and corruption type."""
    filtered = robust_df[robust_df["corruption"].isin(CORRUPTION_DISPLAY)].copy()
    if filtered.empty:
        print("[WARNING] No valid robustness rows for drop heatmap.")
        return

    pivot = (
        filtered.groupby(["model", "corruption"], as_index=False)["drop"]
        .mean()
        .pivot(index="model", columns="corruption", values="drop")
        .reindex(columns=["jpeg", "gaussian_blur", "gaussian_noise"])
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    values = pivot.values.astype(float)
    im = ax.imshow(values, cmap="Reds", aspect="auto")

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([CORRUPTION_DISPLAY.get(c, c) for c in pivot.columns], rotation=15, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text_val = "nan" if np.isnan(values[i, j]) else f"{values[i, j]:.4f}"
            ax.text(j, i, text_val, ha="center", va="center", color="black", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average accuracy drop (clean - corrupted)")
    ax.set_title("CIFAKE Robustness Drop Heatmap by Model and Corruption")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved drop heatmap: {output_path}")


def main() -> None:
    """Entry point for final comparison generation."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    resnet_clean_df = read_csv_if_exists(args.resnet_clean_csv, "ResNet clean CSV")
    clean_df = build_clean_comparison(resnet_clean_df)
    clean_out = args.output_dir / "model_comparison_clean.csv"
    clean_df.to_csv(clean_out, index=False)
    print(f"Saved clean comparison CSV: {clean_out}")

    ml_robust_df = read_csv_if_exists(args.ml_robustness_csv, "ML robustness CSV")
    resnet_robust_df = read_csv_if_exists(args.resnet_robustness_csv, "ResNet robustness CSV")

    robust_parts: list[pd.DataFrame] = []
    for name, frame in [("ML robustness", ml_robust_df), ("ResNet robustness", resnet_robust_df)]:
        if frame is None:
            continue
        try:
            robust_parts.append(normalize_robustness_columns(frame))
        except Exception as exc:
            print(f"[WARNING] Could not use {name} CSV due to format issue: {exc}")

    if not robust_parts:
        print("[WARNING] No valid robustness data available. Robustness plots were not generated.")
        return

    robust_df = pd.concat(robust_parts, ignore_index=True)

    curves_out = args.output_dir / "robustness_curves_all_models.png"
    plot_robustness_curves(robust_df, curves_out)

    heatmap_out = args.output_dir / "drop_heatmap_all_models.png"
    plot_drop_heatmap(robust_df, heatmap_out)


if __name__ == "__main__":
    main()
