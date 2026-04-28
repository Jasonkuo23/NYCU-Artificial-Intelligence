#!/usr/bin/env python3
"""Supervised-from-scratch baseline on CIFAR-10.

This trains the same CIFAR-modified ResNet-18 backbone end-to-end with a
classification head (512 -> 10).

Usage:
  python3 train_supervised.py --data-dir ./data --epochs 200 --batch-size 512
"""

from __future__ import annotations

import argparse
from pathlib import Path
import csv

import torch
from torch import nn
from torch.nn import functional as F

from simclr.dataset import build_cifar10_supervised_dataloaders
from simclr.model import SupervisedResNet18CIFAR10


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    return correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised ResNet-18 baseline (CIFAR-10)")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Optional: write metrics to a CSV file (epoch, loss, test_acc)",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = build_cifar10_supervised_dataloaders(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = SupervisedResNet18CIFAR10().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0

    csv_file = None
    csv_writer = None
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(log_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=["epoch", "loss", "test_acc"]) 
        csv_writer.writeheader()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        avg_loss = running_loss / max(1, num_batches)
        acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, acc)
        print(f"Epoch {epoch:03d}/{args.epochs} - loss: {avg_loss:.4f} - test_acc: {acc*100:.2f}% (best {best_acc*100:.2f}%)")

        if csv_writer is not None:
            csv_writer.writerow({"epoch": epoch, "loss": avg_loss, "test_acc": acc})

    if csv_file is not None:
        csv_file.close()


if __name__ == "__main__":
    main()
