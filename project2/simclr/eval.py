from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_features(
    *,
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features and labels for an entire dataloader.

    - encoder: should map images -> (B, D) features (e.g., SimCLR.backbone)
    - normalize: if True, L2-normalize features (useful for cosine-sim kNN)

    Returns:
      feats: (N, D) float32 on CPU
      labels: (N,) int64 on CPU
    """

    encoder.eval()

    all_feats: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        feats = encoder(images)
        if normalize:
            feats = torch.nn.functional.normalize(feats, dim=1)
        all_feats.append(feats.detach().cpu())
        all_labels.append(labels.detach().cpu())

    feats = torch.cat(all_feats, dim=0).to(torch.float32)
    labels = torch.cat(all_labels, dim=0).to(torch.int64)
    return feats, labels


@torch.no_grad()
def knn_monitor(
    *,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    k: int = 20,
    chunk_size: int = 256,
    weighted: bool = True,
) -> float:
    """kNN classification accuracy using cosine similarity.

    Assumes `train_feats` and `test_feats` are L2-normalized so cosine
    similarity is a dot product.

    This is implemented in chunks to avoid building the full (N_test x N_train)
    similarity matrix.

    Args:
      weighted: if True, vote weights are the (positive) similarities.

    Returns:
      accuracy in [0, 1]
    """

    if k <= 0:
        raise ValueError("k must be positive")

    device = train_feats.device
    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)
    test_feats = test_feats.to(device)
    test_labels = test_labels.to(device)

    num_classes = int(train_labels.max().item() + 1)

    correct = 0
    total = 0

    for start in range(0, test_feats.shape[0], chunk_size):
        end = min(start + chunk_size, test_feats.shape[0])
        feats = test_feats[start:end]  # (C, D)

        # Cosine similarity via matrix multiplication: (C, D) @ (D, N) -> (C, N)
        sims = feats @ train_feats.T
        top_sims, top_idx = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)
        top_labels = train_labels[top_idx]  # (C, k)

        if weighted:
            # Weighted vote by similarity (clamp at 0 so dissimilar neighbors don't vote negatively).
            weights = torch.clamp(top_sims, min=0.0)
            votes = torch.zeros((top_labels.shape[0], num_classes), device=device)
            votes.scatter_add_(dim=1, index=top_labels, src=weights)
            pred = votes.argmax(dim=1)
        else:
            # Unweighted majority vote.
            votes = torch.zeros((top_labels.shape[0], num_classes), device=device)
            ones = torch.ones_like(top_labels, dtype=torch.float32)
            votes.scatter_add_(dim=1, index=top_labels, src=ones)
            pred = votes.argmax(dim=1)

        correct += int((pred == test_labels[start:end]).sum().item())
        total += (end - start)

    return correct / max(1, total)
