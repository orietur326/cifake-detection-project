"""Metric and reporting helpers for CIFAKE binary classification."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)


def compute_accuracy(y_true, y_pred) -> float:
    """Compute classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def compute_f1(y_true, y_pred, average: str = "binary") -> float:
    """Compute F1 score (binary by default for REAL/FAKE classification)."""
    return float(f1_score(y_true, y_pred, average=average))


def save_confusion_matrix_plot(
    y_true,
    y_pred,
    output_path: Path,
    labels: tuple[str, str] = ("REAL", "FAKE"),
    title: str = "Confusion Matrix",
) -> None:
    """Generate and save a confusion matrix plot as PNG."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels))
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def export_classification_report(
    y_true,
    y_pred,
    output_csv_path: Path,
    labels: tuple[str, str] = ("REAL", "FAKE"),
) -> pd.DataFrame:
    """Compute sklearn classification report and export it to CSV."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=list(labels),
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).T
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=True)
    return df
