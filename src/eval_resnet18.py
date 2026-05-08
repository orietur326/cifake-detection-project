"""Evaluate trained ResNet-18 baseline on clean CIFAKE test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import DEFAULT_SEED
from .data import Sample, load_cifake_samples
from .metrics import (
    compute_accuracy,
    compute_f1,
    export_classification_report,
    save_confusion_matrix_plot,
)
from .models import build_resnet18_cifake
from .utils import ensure_dir, set_seed


class CIFAKEDataset(Dataset):
    """Torch Dataset wrapping CIFAKE ``Sample`` entries."""

    def __init__(self, samples: list[Sample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CIFAKE ResNet-18 on clean test split")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to CIFAKE root directory")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/resnet18"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_train_each", type=int, default=None)
    parser.add_argument("--max_test_each", type=int, default=None)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoint_path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    ckpt_path = args.checkpoint_path or (output_dir / "best_resnet18.pth")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, test_samples = load_cifake_samples(
        data_dir=args.data_dir,
        max_train_each=args.max_train_each,
        max_test_each=args.max_test_each,
        seed=args.seed,
        fast_dev_run=args.fast_dev_run,
    )

    if args.fast_dev_run:
        test_samples = test_samples[:16]

    transform = transforms.Compose([transforms.ToTensor()])
    test_ds = CIFAKEDataset(test_samples, transform=transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_resnet18_cifake(num_classes=2)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    acc = compute_accuracy(y_true, y_pred)
    f1 = compute_f1(y_true, y_pred)

    summary = pd.DataFrame([{"accuracy": acc, "f1": f1}])
    summary_path = output_dir / "resnet18_clean_results.csv"
    summary.to_csv(summary_path, index=False)

    report_path = output_dir / "resnet18_classification_report.csv"
    report_df = export_classification_report(y_true, y_pred, report_path)

    cm_path = output_dir / "resnet18_confusion_matrix.png"
    save_confusion_matrix_plot(
        y_true,
        y_pred,
        output_path=cm_path,
        labels=("REAL", "FAKE"),
        title="ResNet-18 Confusion Matrix (Clean Test)",
    )

    print(json.dumps({"accuracy": acc, "f1": f1}, indent=2))
    print(f"Saved summary: {summary_path}")
    print(f"Saved classification report: {report_path}")
    print(f"Saved confusion matrix: {cm_path}")
    print(report_df)


if __name__ == "__main__":
    main()
