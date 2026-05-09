"""Generate ResNet-18 report figures and summary tables from existing outputs only."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.visualizations.plot_resnet_avg_drop import compute_avg_drop, plot_avg_drop
from src.visualizations.plot_resnet_robustness_curves import plot_robustness_curves
from src.visualizations.plot_resnet_training_curves import plot_training_curves
from src.visualizations.visualize_corruptions import generate_corruption_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report assets from existing ResNet-18 outputs")
    parser.add_argument("--data_dir", type=Path, default=Path("data/CIFAKE"))
    parser.add_argument("--resnet_clean_csv", type=Path, default=Path("outputs/resnet18/resnet18_clean_results.csv"))
    parser.add_argument("--resnet_robustness_csv", type=Path, default=Path("outputs/resnet18/resnet18_robustness_results.csv"))
    parser.add_argument("--training_log_csv", type=Path, default=Path("outputs/resnet18/training_log.csv"))
    parser.add_argument("--confusion_matrix_png", type=Path, default=Path("outputs/resnet18/resnet18_confusion_matrix.png"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/resnet18_report_figures"))
    return parser.parse_args()


def _make_clean_summary(clean_csv: Path, output_path: Path) -> tuple[float, float]:
    clean_df = pd.read_csv(clean_csv)
    row = clean_df.iloc[0]
    acc = float(row["accuracy"])
    f1 = float(row["f1"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = ["Accuracy", "F1"]
    values = [acc, f1]
    bars = ax.bar(metrics, values, color=["#1f77b4", "#2ca02c"])
    ax.set_ylim(0, 1.0)
    ax.set_title("ResNet-18 Clean Evaluation Summary")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.4f}", ha="center")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return acc, f1


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clean_fig = args.output_dir / "resnet18_clean_summary.png"
    training_fig = args.output_dir / "resnet18_training_curves.png"
    robust_fig = args.output_dir / "resnet18_robustness_curves.png"
    avg_bar_fig = args.output_dir / "resnet18_avg_drop_bar.png"
    avg_heatmap_fig = args.output_dir / "resnet18_avg_drop_heatmap.png"
    cm_copy = args.output_dir / "resnet18_confusion_matrix_copy.png"
    corruption_fig = args.output_dir / "corruption_examples.png"
    summary_csv = args.output_dir / "resnet18_final_summary_table.csv"

    clean_acc, clean_f1 = _make_clean_summary(args.resnet_clean_csv, clean_fig)
    plot_training_curves(args.training_log_csv, training_fig)
    plot_robustness_curves(args.resnet_robustness_csv, robust_fig)

    avg_df = compute_avg_drop(args.resnet_robustness_csv)
    plot_avg_drop(avg_df, avg_bar_fig, avg_heatmap_fig)

    shutil.copy2(args.confusion_matrix_png, cm_copy)

    if args.data_dir.exists():
        generate_corruption_examples(args.data_dir, corruption_fig, seed=42)
    else:
        print(f"Skipped corruption examples because data_dir does not exist: {args.data_dir}")

    avg_map = {r["corruption"]: float(r["drop"]) for _, r in avg_df.iterrows()}
    final = pd.DataFrame(
        [
            {
                "clean_accuracy": clean_acc,
                "clean_f1": clean_f1,
                "avg_drop_jpeg": avg_map.get("jpeg", float("nan")),
                "avg_drop_gaussian_blur": avg_map.get("gaussian_blur", float("nan")),
                "avg_drop_gaussian_noise": avg_map.get("gaussian_noise", float("nan")),
            }
        ]
    )
    final.to_csv(summary_csv, index=False)

    print("Generated report assets in:", args.output_dir)


if __name__ == "__main__":
    main()
