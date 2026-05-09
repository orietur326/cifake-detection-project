"""Run Part 1 traditional ML baselines for CIFAKE.

Pipeline:
1) Load CIFAKE train/test images.
2) Extract handcrafted features (DCT + FFT radial stats + RGB histogram).
3) Fit StandardScaler on training features only.
4) Train LogisticRegression, LinearSVC, and XGBoost.
5) Evaluate clean test and robustness under JPEG/blur/noise corruptions.
6) Save CSVs and plots for report usage.
"""

from __future__ import annotations

import argparse
import io
import random
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from scipy.fftpack import dct
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover
    XGBClassifier = None
    XGB_IMPORT_ERROR = exc
else:
    XGB_IMPORT_ERROR = None

IMG_SIZE = 32
REAL_LABEL = 0
FAKE_LABEL = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIFAKE traditional ML baselines.")
    parser.add_argument("--data_dir", type=Path, required=True, help="CIFAKE root directory.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/ml_baseline"))
    parser.add_argument("--max_train_each", type=int, default=None)
    parser.add_argument("--max_test_each", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast_dev_run", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _take_paths(folder: Path, max_count: int | None) -> list[Path]:
    paths = sorted(folder.glob("*.jpg"))
    if max_count is not None:
        return paths[:max_count]
    return paths


def load_class_images(folder: Path, label: int, max_count: int | None) -> tuple[list[np.ndarray], np.ndarray]:
    paths = _take_paths(folder, max_count)
    images: list[np.ndarray] = []
    labels = np.full(len(paths), label, dtype=np.int64)
    for path in paths:
        arr = np.array(Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8)
        images.append(arr)
    return images, labels


def extract_dct_features(img: np.ndarray) -> np.ndarray:
    gray = np.mean(img, axis=2).astype(np.float32) / 255.0
    coeff = dct(dct(gray.T, norm="ortho").T, norm="ortho")
    return coeff.flatten()


def extract_fft_features(img: np.ndarray) -> np.ndarray:
    gray = np.mean(img, axis=2).astype(np.float32) / 255.0
    freq = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(freq))

    cy, cx = IMG_SIZE // 2, IMG_SIZE // 2
    y, x = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    n_bins = IMG_SIZE // 2
    feats = np.zeros(n_bins * 2, dtype=np.float32)
    for r in range(n_bins):
        vals = mag[radius == r]
        if len(vals):
            feats[r] = vals.mean()
            feats[r + n_bins] = vals.std()
    return feats


def extract_color_hist(img: np.ndarray, bins: int = 32) -> np.ndarray:
    parts = []
    for c in range(3):
        hist, _ = np.histogram(img[:, :, c], bins=bins, range=(0, 256))
        parts.append(hist / max(hist.sum(), 1))
    return np.concatenate(parts).astype(np.float32)


def extract_features(images: list[np.ndarray]) -> np.ndarray:
    feats = []
    for img in images:
        feats.append(np.concatenate([extract_dct_features(img), extract_fft_features(img), extract_color_hist(img)]))
    return np.asarray(feats, dtype=np.float32)


def corrupt_jpeg(images: list[np.ndarray], quality: int) -> list[np.ndarray]:
    out = []
    for arr in images:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out.append(np.array(Image.open(buf).convert("RGB"), dtype=np.uint8))
    return out


def corrupt_blur(images: list[np.ndarray], sigma: float) -> list[np.ndarray]:
    return [np.array(Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.uint8) for arr in images]


def corrupt_noise(images: list[np.ndarray], sigma: float) -> list[np.ndarray]:
    out = []
    for arr in images:
        noise = np.random.normal(0, sigma * 255, arr.shape)
        noisy = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        out.append(noisy)
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.fast_dev_run:
        max_train_each = 64
        max_test_each = 32
        print("[fast_dev_run] Enabled: using small subset for quick checks.")
    else:
        max_train_each = args.max_train_each
        max_test_each = args.max_test_each

    train_real = args.data_dir / "train" / "REAL"
    train_fake = args.data_dir / "train" / "FAKE"
    test_real = args.data_dir / "test" / "REAL"
    test_fake = args.data_dir / "test" / "FAKE"

    print("[1/7] Loading image paths and images...")
    tr_real, y_tr_real = load_class_images(train_real, REAL_LABEL, max_train_each)
    tr_fake, y_tr_fake = load_class_images(train_fake, FAKE_LABEL, max_train_each)
    te_real, y_te_real = load_class_images(test_real, REAL_LABEL, max_test_each)
    te_fake, y_te_fake = load_class_images(test_fake, FAKE_LABEL, max_test_each)

    x_train_images = tr_real + tr_fake
    y_train = np.concatenate([y_tr_real, y_tr_fake])
    x_test_images = te_real + te_fake
    y_test = np.concatenate([y_te_real, y_te_fake])
    print(f"Loaded train={len(x_train_images)}, test={len(x_test_images)} images")

    print("[2/7] Extracting handcrafted features...")
    x_train = extract_features(x_train_images)
    x_test = extract_features(x_test_images)
    print(f"Feature shapes: train={x_train.shape}, test={x_test.shape}")

    print("[3/7] Fitting StandardScaler on training features only...")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    print("[4/7] Training models...")
    if XGBClassifier is None:
        raise ImportError(f"xgboost is required for this script: {XGB_IMPORT_ERROR}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=args.seed, n_jobs=-1),
        "LinearSVC": CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0, random_state=args.seed)),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=args.seed,
            n_jobs=-1,
            verbosity=0,
        ),
    }

    clean_records = []
    preds_clean: dict[str, np.ndarray] = {}
    for name, model in models.items():
        print(f"  - training {name}")
        x_train_used = x_train_scaled if name != "XGBoost" else x_train
        x_test_used = x_test_scaled if name != "XGBoost" else x_test
        model.fit(x_train_used, y_train)
        preds = model.predict(x_test_used)
        preds_clean[name] = preds
        clean_records.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, preds),
                "f1": f1_score(y_test, preds),
            }
        )

    print("[5/7] Evaluating clean test set and plotting confusion matrices...")
    clean_df = pd.DataFrame(clean_records).sort_values("model")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, name in zip(axes, models.keys()):
        cm = confusion_matrix(y_test, preds_clean[name], labels=[REAL_LABEL, FAKE_LABEL])
        ConfusionMatrixDisplay(cm, display_labels=["REAL", "FAKE"]).plot(ax=ax, colorbar=False)
        acc = clean_df.loc[clean_df["model"] == name, "accuracy"].iloc[0]
        ax.set_title(f"{name}\nacc={acc:.4f}")

    print("[6/7] Evaluating robustness under JPEG / blur / noise...")
    corruptions: dict[str, list[tuple[Callable, float | int, str]]] = {
        "JPEG quality": [(corrupt_jpeg, 75, "q=75"), (corrupt_jpeg, 50, "q=50"), (corrupt_jpeg, 25, "q=25")],
        "Gaussian blur": [(corrupt_blur, 1, "σ=1"), (corrupt_blur, 2, "σ=2"), (corrupt_blur, 3, "σ=3")],
        "Gaussian noise": [
            (corrupt_noise, 0.05, "σ=0.05"),
            (corrupt_noise, 0.10, "σ=0.10"),
            (corrupt_noise, 0.20, "σ=0.20"),
        ],
    }

    clean_acc = {r["model"]: r["accuracy"] for r in clean_records}
    robustness_rows = []

    for corruption_name, levels in corruptions.items():
        print(f"  - {corruption_name}")
        for fn, param, label in levels:
            x_corrupt = extract_features(fn(x_test_images, param))
            x_corrupt_scaled = scaler.transform(x_corrupt)
            for name, model in models.items():
                x_used = x_corrupt_scaled if name != "XGBoost" else x_corrupt
                acc = accuracy_score(y_test, model.predict(x_used))
                drop = clean_acc[name] - acc
                robustness_rows.append(
                    {
                        "model": name,
                        "corruption": corruption_name,
                        "level": label,
                        "accuracy": acc,
                        "drop": drop,
                    }
                )

    robustness_df = pd.DataFrame(robustness_rows)
    avg_drop_df = (
        robustness_df.groupby(["model", "corruption"], as_index=False)["drop"]
        .mean()
        .rename(columns={"drop": "avg_drop"})
        .sort_values(["model", "corruption"])
    )

    print("[7/7] Saving outputs...")
    output_dir = args.output_dir
    report_data_dir = Path("report/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_data_dir.mkdir(parents=True, exist_ok=True)

    clean_path = output_dir / "ml_clean_results.csv"
    robustness_path = output_dir / "ml_robustness_results.csv"
    avg_drop_path = output_dir / "ml_avg_robustness_drop.csv"
    cm_path = output_dir / "confusion_matrices_clean.png"
    robustness_plot_path = output_dir / "robustness_curves.png"
    heatmap_path = output_dir / "drop_heatmap.png"

    clean_df.to_csv(clean_path, index=False)
    robustness_df.to_csv(robustness_path, index=False)
    avg_drop_df.to_csv(avg_drop_path, index=False)
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, corruption_name in zip(axes2, corruptions.keys()):
        levels = [x[2] for x in corruptions[corruption_name]]
        for name in models.keys():
            sub = robustness_df[(robustness_df["model"] == name) & (robustness_df["corruption"] == corruption_name)]
            series = sub.set_index("level").loc[levels, "accuracy"].values
            ax.plot(levels, series, marker="o", label=name)
            ax.axhline(clean_acc[name], linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(corruption_name)
        ax.set_xlabel("Corruption level")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(robustness_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    pivot = robustness_df.pivot_table(index="model", columns=["corruption", "level"], values="drop")
    fig3, ax3 = plt.subplots(figsize=(13, 3))
    im = ax3.imshow(pivot.values, cmap="Reds", aspect="auto", vmin=0)
    ax3.set_xticks(range(len(pivot.columns)))
    ax3.set_xticklabels([f"{a}\n{b}" for a, b in pivot.columns], fontsize=8)
    ax3.set_yticks(range(len(pivot.index)))
    ax3.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            ax3.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
    fig3.colorbar(im, ax=ax3, label="Accuracy drop")
    fig3.tight_layout()
    fig3.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    clean_df.to_csv(report_data_dir / "ml_clean_results.csv", index=False)
    avg_drop_df.to_csv(report_data_dir / "ml_avg_robustness_drop.csv", index=False)

    print(f"Saved: {clean_path}")
    print(f"Saved: {robustness_path}")
    print(f"Saved: {avg_drop_path}")


if __name__ == "__main__":
    main()
