"""Evaluate trained ResNet-18 on clean and corrupted CIFAKE test images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import DEFAULT_SEED
from .corruptions import (
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_jpeg_compression,
    corruption_grid,
)
from .data import Sample, load_cifake_samples
from .metrics import compute_accuracy, compute_f1
from .models import build_resnet18_cifake
from .utils import ensure_dir, set_seed


class CIFAKEDataset(Dataset):
    """Torch Dataset wrapping CIFAKE ``Sample`` entries with optional corruption."""

    def __init__(
        self,
        samples: list[Sample],
        *,
        transform=None,
        corruption_fn: Callable[[Image.Image], Image.Image] | None = None,
    ):
        self.samples = samples
        self.transform = transform
        self.corruption_fn = corruption_fn

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        if self.corruption_fn is not None:
            image = self.corruption_fn(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained CIFAKE ResNet-18 under clean and corruption settings"
    )
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to CIFAKE root directory")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/resnet18/best_resnet18.pth"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/resnet18"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_test_each", type=int, default=None)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def evaluate_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Evaluate one loader and return (accuracy, f1)."""
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    acc = compute_accuracy(y_true, y_pred)
    f1 = compute_f1(y_true, y_pred)
    return acc, f1


def build_loader(
    samples: list[Sample],
    *,
    batch_size: int,
    num_workers: int,
    corruption_fn: Callable[[Image.Image], Image.Image] | None,
) -> DataLoader:
    """Create deterministic test loader for clean or corrupted samples."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CIFAKEDataset(samples, transform=transform, corruption_fn=corruption_fn)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    output_dir = ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, test_samples = load_cifake_samples(
        data_dir=args.data_dir,
        max_test_each=args.max_test_each,
        seed=args.seed,
        fast_dev_run=args.fast_dev_run,
    )

    model = build_resnet18_cifake(num_classes=2)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    clean_loader = build_loader(
        test_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        corruption_fn=None,
    )
    clean_acc, clean_f1 = evaluate_loader(model, clean_loader, device)

    rows: list[dict] = [
        {
            "model": "resnet18",
            "corruption": "clean",
            "level": "none",
            "clean_acc": clean_acc,
            "acc": clean_acc,
            "drop": 0.0,
            "f1": clean_f1,
        }
    ]

    grid = corruption_grid()

    for quality in grid["jpeg"]:
        loader = build_loader(
            test_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            corruption_fn=lambda image, q=quality: apply_jpeg_compression(image, q),
        )
        acc, f1 = evaluate_loader(model, loader, device)
        rows.append(
            {
                "model": "resnet18",
                "corruption": "jpeg",
                "level": quality,
                "clean_acc": clean_acc,
                "acc": acc,
                "drop": clean_acc - acc,
                "f1": f1,
            }
        )

    for sigma in grid["gaussian_blur"]:
        loader = build_loader(
            test_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            corruption_fn=lambda image, s=sigma: apply_gaussian_blur(image, s),
        )
        acc, f1 = evaluate_loader(model, loader, device)
        rows.append(
            {
                "model": "resnet18",
                "corruption": "gaussian_blur",
                "level": sigma,
                "clean_acc": clean_acc,
                "acc": acc,
                "drop": clean_acc - acc,
                "f1": f1,
            }
        )

    for sigma in grid["gaussian_noise"]:
        loader = build_loader(
            test_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            corruption_fn=lambda image, s=sigma: apply_gaussian_noise(image, s),
        )
        acc, f1 = evaluate_loader(model, loader, device)
        rows.append(
            {
                "model": "resnet18",
                "corruption": "gaussian_noise",
                "level": sigma,
                "clean_acc": clean_acc,
                "acc": acc,
                "drop": clean_acc - acc,
                "f1": f1,
            }
        )

    result_df = pd.DataFrame(rows, columns=["model", "corruption", "level", "clean_acc", "acc", "drop", "f1"])
    output_csv = output_dir / "resnet18_robustness_results.csv"
    result_df.to_csv(output_csv, index=False)

    print(json.dumps({"clean_acc": clean_acc, "clean_f1": clean_f1}, indent=2))
    print(result_df)
    print(f"Saved robustness results: {output_csv}")


if __name__ == "__main__":
    main()
