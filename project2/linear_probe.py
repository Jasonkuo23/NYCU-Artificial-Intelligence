#!/usr/bin/env python3
"""Linear probing evaluation for SimCLR representations.

This trains a linear classifier on top of a frozen encoder (backbone output, 512-d).

Project requirement defaults:
- Adam optimizer
- lr=1e-3
- weight_decay=1e-6
- epochs=100

Usage:
  # Linear probing after SSL pretraining
  python3 linear_probe.py --data-dir ./data --checkpoint ./checkpoints/simclr_last.pt

  # "Random frozen" lower bound (no checkpoint; encoder is random)
  python3 linear_probe.py --data-dir ./data
"""

from __future__ import annotations

import argparse
from pathlib import Path
import csv

import torch
from torch import nn
from torch.nn import functional as F

from simclr.dataset import build_cifar10_eval_dataloaders
from simclr.model import SimCLR


@torch.no_grad()
def evaluate(encoder: nn.Module, classifier: nn.Module, loader, device: torch.device) -> float:
    encoder.eval()
    classifier.eval()

    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        feats = encoder(images)
        logits = classifier(feats)
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())

    return correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear probing on CIFAR-10")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to SimCLR checkpoint (.pt)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="Optional: cap number of training batches per epoch (debug speed-up)",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Optional: write metrics to a CSV file (epoch, loss, test_acc)",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=0,
        help="Print progress every N batches (0 disables)",
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

    # Build encoder (ResNet-18 CIFAR). If checkpoint provided, load SSL weights.
    simclr = SimCLR(embedding_dim=128)
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        simclr.load_state_dict(ckpt["model"], strict=True)

    encoder = simclr.backbone
    encoder.to(device)

    # Freeze encoder.
    for p in encoder.parameters():
        p.requires_grad = False

    classifier = nn.Linear(512, 10).to(device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader, test_loader = build_cifar10_eval_dataloaders(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

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
        classifier.train()

        running_loss = 0.0
        num_batches = 0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                feats = encoder(images)

            logits = classifier(feats)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

            if args.print_every and (step % args.print_every == 0):
                avg = running_loss / max(1, num_batches)
                print(
                    f"[epoch {epoch:03d}/{args.epochs}] step {step:04d} - batch_loss: {loss.item():.4f} - avg_loss: {avg:.4f}"
                )

            if args.max_steps_per_epoch is not None and step >= args.max_steps_per_epoch:
                break

        avg_loss = running_loss / max(1, num_batches)
        acc = evaluate(encoder, classifier, test_loader, device)
        best_acc = max(best_acc, acc)
        print(f"Epoch {epoch:03d}/{args.epochs} - loss: {avg_loss:.4f} - test_acc: {acc*100:.2f}% (best {best_acc*100:.2f}%)")

        if csv_writer is not None:
            csv_writer.writerow({"epoch": epoch, "loss": avg_loss, "test_acc": acc})

    if csv_file is not None:
        csv_file.close()


if __name__ == "__main__":
    main()
