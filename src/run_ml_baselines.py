"""Reproducible Part 1 traditional ML baselines for CIFAKE.

Pipeline:
- Load CIFAKE train/test images
- Extract handcrafted features (DCT + FFT radial + RGB histogram)
- Train Logistic Regression, LinearSVC, and XGBoost
- Evaluate clean and corrupted robustness performance
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm
from xgboost import XGBClassifier

from .config import DEFAULT_SEED, GAUSSIAN_BLUR_SIGMAS, GAUSSIAN_NOISE_SIGMAS, JPEG_QUALITIES
from .corruptions import apply_gaussian_blur, apply_gaussian_noise, apply_jpeg_compression
from .data import load_cifake_samples
from .utils import ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIFAKE Part 1 traditional ML baselines.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to CIFAKE root directory.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/ml_baseline"))
    parser.add_argument("--max_train_each", type=int, default=None)
    parser.add_argument("--max_test_each", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fast_dev_run", action="store_true")
    return parser.parse_args()


def load_rgb_array(image_path: Path, transform: Callable[[Image.Image], Image.Image] | None = None) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    if transform is not None:
        image = transform(image)
    return np.asarray(image, dtype=np.float32)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return (0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]).astype(np.float32)


def dct_features(rgb: np.ndarray) -> np.ndarray:
    gray = rgb_to_gray(rgb)
    n = gray.shape[0]
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    basis = np.cos(np.pi * (2 * i + 1) * k / (2 * n)).astype(np.float32)
    alpha = np.ones((n, 1), dtype=np.float32) * np.sqrt(2.0 / n)
    alpha[0, 0] = np.sqrt(1.0 / n)
    c = alpha * basis
    coeffs = c @ gray @ c.T
    return coeffs.flatten()  # 32*32 = 1024


def fft_radial_features(rgb: np.ndarray, n_bins: int = 32) -> np.ndarray:
    gray = rgb_to_gray(rgb)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2

    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = np.clip(r.astype(np.int32), 0, n_bins - 1)

    radial_sum = np.bincount(r.ravel(), weights=power.ravel(), minlength=n_bins)
    radial_count = np.bincount(r.ravel(), minlength=n_bins)
    radial_mean = radial_sum / np.maximum(radial_count, 1)
    radial_log = np.log1p(radial_mean)
    return radial_log.astype(np.float32)  # 32 dims


def rgb_hist_features(rgb: np.ndarray, bins: int = 32) -> np.ndarray:
    feats = []
    for c in range(3):
        hist, _ = np.histogram(rgb[:, :, c], bins=bins, range=(0, 256), density=True)
        feats.append(hist.astype(np.float32))
    return np.concatenate(feats, axis=0)  # 96 dims


def extract_features_for_paths(paths: list[Path], transform: Callable[[Image.Image], Image.Image] | None = None) -> np.ndarray:
    rows = []
    for p in tqdm(paths, desc="extract_features", leave=False):
        rgb = load_rgb_array(p, transform=transform)
        feats = np.concatenate([dct_features(rgb), fft_radial_features(rgb), rgb_hist_features(rgb)])
        rows.append(feats)
    return np.vstack(rows).astype(np.float32)


def build_models(seed: int) -> dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=seed),
        "LinearSVC": LinearSVC(random_state=seed, max_iter=5000),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def evaluate_clean(models: dict[str, object], x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
    clean_rows = []
    preds = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        preds[name] = y_pred
        clean_rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred, average="binary"),
            }
        )
    return pd.DataFrame(clean_rows), preds


def plot_confusion_matrices(y_true: np.ndarray, preds: dict[str, np.ndarray], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(preds), figsize=(15, 4))
    for ax, (name, y_pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["REAL", "FAKE"]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def evaluate_robustness(models, scaler, test_paths, y_test, clean_acc_by_model):
    settings = {
        "jpeg": JPEG_QUALITIES,
        "gaussian_blur": GAUSSIAN_BLUR_SIGMAS,
        "gaussian_noise": GAUSSIAN_NOISE_SIGMAS,
    }
    transforms = {
        "jpeg": lambda lvl: (lambda img: apply_jpeg_compression(img, int(lvl))),
        "gaussian_blur": lambda lvl: (lambda img: apply_gaussian_blur(img, float(lvl))),
        "gaussian_noise": lambda lvl: (lambda img: apply_gaussian_noise(img, float(lvl))),
    }

    rows = []
    for corr_name, levels in settings.items():
        for level in levels:
            print(f"Evaluating corruption={corr_name} level={level}")
            x_corr = extract_features_for_paths(test_paths, transform=transforms[corr_name](level))
            x_corr = scaler.transform(x_corr)
            for model_name, model in models.items():
                y_pred = model.predict(x_corr)
                acc = accuracy_score(y_test, y_pred)
                rows.append(
                    {
                        "model": model_name,
                        "corruption": corr_name,
                        "severity": level,
                        "accuracy": acc,
                        "f1": f1_score(y_test, y_pred, average="binary"),
                        "clean_accuracy": clean_acc_by_model[model_name],
                        "accuracy_drop": clean_acc_by_model[model_name] - acc,
                    }
                )
    return pd.DataFrame(rows)


def plot_robustness_curves(df_robust: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for ax, corr in zip(axes, ["jpeg", "gaussian_blur", "gaussian_noise"]):
        d = df_robust[df_robust["corruption"] == corr].copy()
        for model in d["model"].unique():
            dm = d[d["model"] == model].sort_values("severity")
            ax.plot(dm["severity"], dm["accuracy"], marker="o", label=model)
        ax.set_title(corr)
        ax.set_xlabel("severity")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("accuracy")
    axes[-1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_drop_heatmap(df_robust: pd.DataFrame, output_path: Path) -> None:
    df = df_robust.copy()
    df["setting"] = df["corruption"] + "_" + df["severity"].astype(str)
    piv = df.pivot_table(index="model", columns="setting", values="accuracy_drop")
    plt.figure(figsize=(12, 4))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="Reds")
    plt.title("Accuracy Drop vs Clean")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.output_dir)

    train_samples, test_samples = load_cifake_samples(
        args.data_dir,
        max_train_each=args.max_train_each,
        max_test_each=args.max_test_each,
        seed=args.seed,
        fast_dev_run=args.fast_dev_run,
    )
    if args.fast_dev_run:
        train_samples = train_samples[: min(40, len(train_samples))]
        test_samples = test_samples[: min(40, len(test_samples))]

    train_paths = [s.image_path for s in train_samples]
    y_train = np.array([s.label for s in train_samples], dtype=np.int64)
    test_paths = [s.image_path for s in test_samples]
    y_test = np.array([s.label for s in test_samples], dtype=np.int64)

    print("Extracting clean train features...")
    x_train = extract_features_for_paths(train_paths)
    print("Extracting clean test features...")
    x_test = extract_features_for_paths(test_paths)

    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    x_test_sc = scaler.transform(x_test)

    models = build_models(args.seed)
    clean_df, preds = evaluate_clean(models, x_train_sc, y_train, x_test_sc, y_test)
    clean_csv = out_dir / "ml_clean_results.csv"
    clean_df.to_csv(clean_csv, index=False)
    plot_confusion_matrices(y_test, preds, out_dir / "confusion_matrices_clean.png")

    clean_acc = {row["model"]: row["accuracy"] for _, row in clean_df.iterrows()}
    robust_df = evaluate_robustness(models, scaler, test_paths, y_test, clean_acc)
    robust_df.to_csv(out_dir / "ml_robustness_results.csv", index=False)
    plot_robustness_curves(robust_df, out_dir / "robustness_curves.png")
    plot_drop_heatmap(robust_df, out_dir / "drop_heatmap.png")

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
