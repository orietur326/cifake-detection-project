"""Train ResNet-18 baseline for CIFAKE binary classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .config import DEFAULT_SEED
from .data import Sample, load_cifake_samples, split_train_val
from .metrics import compute_accuracy, compute_f1
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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    """Evaluate model on a dataloader and return loss/accuracy/f1."""
    model.eval()
    losses = []
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            losses.append(loss.item())

            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    avg_loss = float(sum(losses) / max(1, len(losses)))
    acc = compute_accuracy(y_true, y_pred)
    f1 = compute_f1(y_true, y_pred)
    return {"loss": avg_loss, "accuracy": acc, "f1": f1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAKE ResNet-18 baseline")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to CIFAKE root directory")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/resnet18"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_train_each", type=int, default=None)
    parser.add_argument("--max_test_each", type=int, default=None)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.fast_dev_run:
        if args.max_train_each is None:
            args.max_train_each = 20
        if args.max_test_each is None:
            args.max_test_each = 20
        args.epochs = 1
        args.batch_size = 8
        args.num_workers = 0
        print(
            "fast_dev_run enabled with overrides: "
            f"max_train_each={args.max_train_each}, max_test_each={args.max_test_each}, "
            f"epochs={args.epochs}, batch_size={args.batch_size}, num_workers={args.num_workers}"
        )

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_samples, test_samples = load_cifake_samples(
        data_dir=args.data_dir,
        max_train_each=args.max_train_each,
        max_test_each=args.max_test_each,
        seed=args.seed,
        fast_dev_run=args.fast_dev_run,
    )
    train_split, val_split = split_train_val(train_samples, val_size=0.2, seed=args.seed)

    print(
        f"Split sample counts -> train: {len(train_split)}, val: {len(val_split)}, test: {len(test_samples)}"
    )

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = CIFAKEDataset(train_split, transform=transform)
    val_ds = CIFAKEDataset(val_split, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"DataLoader config -> batch_size={args.batch_size}, num_workers={args.num_workers}, "
        f"train_batches={len(train_loader)}, val_batches={len(val_loader)}"
    )

    model = build_resnet18_cifake(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_f1 = -1.0
    log_rows: list[dict] = []
    ckpt_path = output_dir / "best_resnet18.pth"

    print(f"Starting training loop. First epoch will be: 1/{args.epochs}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        y_true_train, y_pred_train = [], []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            y_true_train.extend(labels.detach().cpu().tolist())
            y_pred_train.extend(preds.detach().cpu().tolist())

        train_loss = float(running_loss / max(1, len(train_loader)))
        train_acc = compute_accuracy(y_true_train, y_pred_train)
        train_f1 = compute_f1(y_true_train, y_pred_train)

        val_metrics = evaluate(model, val_loader, device, criterion)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
        }
        log_rows.append(row)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, train_f1={train_f1:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.4f}, val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved improved checkpoint to: {ckpt_path}")

    log_df = pd.DataFrame(log_rows)
    log_path = output_dir / "training_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"Saved training log: {log_path}")


if __name__ == "__main__":
    main()
