from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NTXentLoss(nn.Module):
    """Normalized Temperature-Scaled Cross Entropy (NT-Xent) loss.

    Given two batches of embeddings (z1, z2) corresponding to two augmented
    views of the same images:
      - (z1[i], z2[i]) is a positive pair
      - all other combinations in the batch are negatives

    Key idea:
    - Use cosine similarity (dot product after L2-normalization)
    - For each anchor, classify its matching positive among (2B - 1) candidates

    Efficient implementation:
    - Concatenate embeddings: z = [z1; z2]  -> shape (2B, D)
    - Pairwise cosine similarity is z @ z^T because embeddings are normalized
    - Apply temperature scaling by dividing logits by tau

    Loss is the mean cross-entropy over all 2B anchors.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.ndim != 2 or z2.ndim != 2:
            raise ValueError("z1 and z2 must be 2D tensors of shape (B, D)")
        if z1.shape != z2.shape:
            raise ValueError("z1 and z2 must have the same shape")

        batch_size = z1.shape[0]
        if batch_size <= 1:
            raise ValueError("batch_size must be > 1 for contrastive loss")

        # Ensure embeddings are normalized (model already does this, but keep robust).
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)  # (2B, D)

        # Pairwise cosine similarity via matrix multiplication.
        # Since z is L2-normalized, cosine_similarity(z_i, z_j) == z_i @ z_j.
        logits = (z @ z.T) / self.temperature  # (2B, 2B)

        # Remove similarity of samples with themselves.
        logits.fill_diagonal_(float("-inf"))

        # Positive for i is i+B (first view <-> second view), and vice versa.
        labels = torch.arange(2 * batch_size, device=z.device)
        labels = (labels + batch_size) % (2 * batch_size)

        # Cross-entropy expects class scores over all candidates.
        # With -inf on the diagonal, the model cannot select itself.
        loss = F.cross_entropy(logits, labels)
        return loss
