#!/usr/bin/env python3
"""SimCLR baseline training on CIFAR-10.

This script is intentionally simple and educational:
- CIFAR-10 dataset with SimCLR-style augmentations producing two views.
- ResNet-18 backbone modified for CIFAR-10 (3x3 conv1, no maxpool, no FC).
- MLP projector head (512 -> 512 -> 128).
- NT-Xent contrastive loss using cosine similarity + temperature.

Run:
  python train_simclr.py --data-dir ./data --epochs 200 --batch-size 512

Notes:
- This trains *self-supervised* representations; it does not use labels.
- Kept modular so you can later add kNN monitor / linear probing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import csv

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from simclr.dataset import build_cifar10_eval_dataloaders, build_cifar10_simclr_dataloader
from simclr.eval import extract_features, knn_monitor
from simclr.loss import NTXentLoss
from simclr.model import SimCLR


@dataclass
class TrainConfig:
    data_dir: str
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    temperature: float
    num_workers: int
    seed: int
    device: str
    knn_k: int
    knn_every: int
    eval_batch_size: int
    knn_chunk_size: int
    save_path: str | None
    knn_max_train: int | None
    knn_max_test: int | None
    max_steps_per_epoch: int | None
    log_path: str | None
    print_every: int


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int | None = None,
    print_every: int = 0,
    epoch: int | None = None,
    total_epochs: int | None = None,
) -> float:
    model.train()

    running_loss = 0.0
    num_batches = 0

    for step, (x1, x2) in enumerate(loader, start=1):
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        z1 = model(x1)
        z2 = model(x2)

        loss = criterion(z1, z2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        if print_every and (step % print_every == 0):
            avg = running_loss / max(1, num_batches)
            prefix = ""
            if epoch is not None and total_epochs is not None:
                prefix = f"[epoch {epoch:03d}/{total_epochs}] "
            print(f"{prefix}step {step:04d} - batch_loss: {loss.item():.4f} - avg_loss: {avg:.4f}")

        if max_steps is not None and step >= max_steps:
            break

    return running_loss / max(1, num_batches)


def main() -> None:
    parser = argparse.ArgumentParser(description="SimCLR baseline (CIFAR-10, PyTorch)")
    parser.add_argument("--data-dir", type=str, default="./data", help="Where CIFAR-10 is/will be stored")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--knn-every", type=int, default=5, help="Run kNN monitor every N epochs (0 disables)")
    parser.add_argument("--eval-batch-size", type=int, default=512, help="Batch size for kNN feature extraction")
    parser.add_argument("--knn-chunk-size", type=int, default=256, help="Chunk size for kNN similarity computation")
    parser.add_argument(
        "--knn-max-train",
        type=int,
        default=None,
        help="Optional: use only first N train samples for kNN (debug speed-up)",
    )
    parser.add_argument(
        "--knn-max-test",
        type=int,
        default=None,
        help="Optional: use only first N test samples for kNN (debug speed-up)",
    )
    parser.add_argument("--save-path", type=str, default="./checkpoints/simclr_last.pt")
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="Optional: cap number of training batches per epoch (debug speed-up)",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Optional: write metrics to a CSV file (epoch, loss, knn_acc)",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=0,
        help="Print progress every N batches (0 disables)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        knn_k=args.knn_k,
        knn_every=args.knn_every,
        eval_batch_size=args.eval_batch_size,
        knn_chunk_size=args.knn_chunk_size,
        save_path=args.save_path,
        knn_max_train=args.knn_max_train,
        knn_max_test=args.knn_max_test,
        max_steps_per_epoch=args.max_steps_per_epoch,
        log_path=args.log_path,
        print_every=args.print_every,
    )

    set_seed(cfg.seed)

    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)

    loader = build_cifar10_simclr_dataloader(
        data_dir=str(data_dir),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # Dataloaders for kNN monitor use deterministic (no-augmentation) transforms.
    eval_train_loader, eval_test_loader = build_cifar10_eval_dataloaders(
        data_dir=str(data_dir),
        batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
    )

    if cfg.knn_max_train is not None:
        eval_train_loader = DataLoader(
            Subset(eval_train_loader.dataset, range(min(cfg.knn_max_train, len(eval_train_loader.dataset)))),
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    if cfg.knn_max_test is not None:
        eval_test_loader = DataLoader(
            Subset(eval_test_loader.dataset, range(min(cfg.knn_max_test, len(eval_test_loader.dataset)))),
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    model = SimCLR(embedding_dim=128)
    model = model.to(device)

    criterion = NTXentLoss(temperature=cfg.temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    csv_file = None
    csv_writer = None
    if cfg.log_path:
        log_path = Path(cfg.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(log_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=["epoch", "loss", "knn_acc"]) 
        csv_writer.writeheader()

    for epoch in range(1, cfg.epochs + 1):
        avg_loss = train_one_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_steps=cfg.max_steps_per_epoch,
            print_every=cfg.print_every,
            epoch=epoch,
            total_epochs=cfg.epochs,
        )

        msg = f"Epoch {epoch:03d}/{cfg.epochs} - loss: {avg_loss:.4f}"
        knn_acc = None

        # kNN monitor: uses backbone representations (512-d), not projector outputs.
        if cfg.knn_every > 0 and (epoch % cfg.knn_every == 0 or epoch == cfg.epochs):
            with torch.no_grad():
                train_feats, train_labels = extract_features(
                    encoder=model.backbone,
                    loader=eval_train_loader,
                    device=device,
                    normalize=True,
                )
                test_feats, test_labels = extract_features(
                    encoder=model.backbone,
                    loader=eval_test_loader,
                    device=device,
                    normalize=True,
                )

            acc = knn_monitor(
                train_feats=train_feats,
                train_labels=train_labels,
                test_feats=test_feats,
                test_labels=test_labels,
                k=cfg.knn_k,
                chunk_size=cfg.knn_chunk_size,
                weighted=True,
            )
            knn_acc = acc
            msg += f" - knn@{cfg.knn_k}: {acc*100:.2f}%"

        print(msg)

        if csv_writer is not None:
            csv_writer.writerow(
                {
                    "epoch": epoch,
                    "loss": avg_loss,
                    "knn_acc": "" if knn_acc is None else knn_acc,
                }
            )

    if csv_file is not None:
        csv_file.close()

    # Save final checkpoint for linear probing.
    if cfg.save_path:
        save_path = Path(cfg.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
            },
            str(save_path),
        )
        print(f"Saved checkpoint to: {save_path}")


if __name__ == "__main__":
    main()
