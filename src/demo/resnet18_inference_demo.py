"""Practical inference demo for a trained CIFAKE ResNet-18 checkpoint."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

from src.corruptions import apply_gaussian_blur, apply_gaussian_noise, apply_jpeg_compression
from src.data import Sample, load_cifake_samples
from src.models import build_resnet18_cifake
from src.utils import ensure_dir, set_seed

IDX_TO_LABEL = {0: "REAL", 1: "FAKE"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the inference demo."""
    parser = argparse.ArgumentParser(description="Run practical CIFAKE ResNet-18 inference demo")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to CIFAKE root directory")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/resnet18/best_resnet18.pth"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/resnet18_report_figures"))
    parser.add_argument("--num_examples", type=int, default=8, help="Total examples for mixed REAL/FAKE grid")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    """Load trained CIFAKE ResNet-18 from checkpoint."""
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = build_resnet18_cifake(num_classes=2)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_pil(model: torch.nn.Module, image: Image.Image, device: torch.device) -> tuple[int, float]:
    """Run one-image prediction and return predicted class plus confidence."""
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return int(pred.item()), float(conf.item())


def sample_real_fake(test_samples: list[Sample], num_examples: int, seed: int) -> list[Sample]:
    """Randomly pick a balanced list of REAL and FAKE examples from test split."""
    rng = random.Random(seed)
    real = [s for s in test_samples if s.label == 0]
    fake = [s for s in test_samples if s.label == 1]
    half = max(1, num_examples // 2)
    k_real = min(len(real), half)
    k_fake = min(len(fake), num_examples - k_real)
    picks = rng.sample(real, k=k_real) + rng.sample(fake, k=k_fake)
    rng.shuffle(picks)
    return picks


def save_inference_examples(model, samples: list[Sample], device: torch.device, out_path: Path) -> None:
    """Save mixed REAL/FAKE inference examples figure."""
    cols = 4
    rows = max(1, (len(samples) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.8 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, sample in zip(axes, samples):
        image = Image.open(sample.image_path).convert("RGB")
        pred, conf = predict_pil(model, image, device)
        gt_label = IDX_TO_LABEL[sample.label]
        pred_label = IDX_TO_LABEL[pred]
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"GT: {gt_label} | Pred: {pred_label}\nConf: {conf:.3f}", fontsize=10)

    for ax in axes[len(samples) :]:
        ax.axis("off")

    fig.suptitle("ResNet-18 Practical Inference on CIFAKE Test Images", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_corruption_demo(model, real_sample: Sample, fake_sample: Sample, device: torch.device, out_path: Path) -> None:
    """Save corruption sensitivity visualization for one REAL and one FAKE sample."""
    conditions = [
        ("Original", lambda img: img),
        ("JPEG q=25", lambda img: apply_jpeg_compression(img, quality=25)),
        ("Blur σ=3", lambda img: apply_gaussian_blur(img, sigma=3)),
        ("Noise σ=0.2", lambda img: apply_gaussian_noise(img, sigma=0.2)),
    ]
    selected = [real_sample, fake_sample]
    fig, axes = plt.subplots(len(selected), len(conditions), figsize=(4 * len(conditions), 4 * len(selected)))

    for r, sample in enumerate(selected):
        base = Image.open(sample.image_path).convert("RGB")
        gt_label = IDX_TO_LABEL[sample.label]
        for c, (name, fn) in enumerate(conditions):
            img = fn(base)
            pred, conf = predict_pil(model, img, device)
            pred_label = IDX_TO_LABEL[pred]
            ax = axes[r, c] if len(selected) > 1 else axes[c]
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{name}\nGT: {gt_label} Pred: {pred_label}\nConf: {conf:.3f}", fontsize=10)

    fig.suptitle("Corruption Sensitivity Demo (REAL and FAKE)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_failure_cases(model, test_samples: list[Sample], device: torch.device, out_path: Path, seed: int) -> None:
    """Collect and visualize several misclassified test examples if available."""
    rng = random.Random(seed)
    subset = test_samples[: min(len(test_samples), 400)]
    misclassified: list[tuple[Sample, int, float]] = []

    for sample in subset:
        image = Image.open(sample.image_path).convert("RGB")
        pred, conf = predict_pil(model, image, device)
        if pred != sample.label:
            misclassified.append((sample, pred, conf))

    if not misclassified:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No misclassified examples found in inspected subset.", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    show = rng.sample(misclassified, k=min(8, len(misclassified)))
    cols = 4
    rows = (len(show) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (sample, pred, conf) in zip(axes, show):
        image = Image.open(sample.image_path).convert("RGB")
        gt_label = IDX_TO_LABEL[sample.label]
        pred_label = IDX_TO_LABEL[pred]
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"GT: {gt_label} | Pred: {pred_label}\nConf: {conf:.3f}", fontsize=10)

    for ax in axes[len(show) :]:
        ax.axis("off")

    fig.suptitle("Failure Cases (Misclassified Test Images)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_samples = load_cifake_samples(data_dir=args.data_dir, seed=args.seed)
    if len(test_samples) < 2:
        raise ValueError("Not enough test samples found for demo.")

    model = load_model(args.checkpoint, device)

    mixed_samples = sample_real_fake(test_samples, num_examples=args.num_examples, seed=args.seed)
    save_inference_examples(model, mixed_samples, device, output_dir / "inference_examples.png")

    real_sample = next(s for s in test_samples if s.label == 0)
    fake_sample = next(s for s in test_samples if s.label == 1)
    save_corruption_demo(model, real_sample, fake_sample, device, output_dir / "corrupted_inference_examples.png")

    save_failure_cases(model, test_samples, device, output_dir / "failure_cases.png", seed=args.seed)

    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
