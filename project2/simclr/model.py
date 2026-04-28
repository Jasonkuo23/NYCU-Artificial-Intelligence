from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


def build_resnet18_cifar10() -> nn.Module:
    """ResNet-18 backbone modified for CIFAR-10.

    Changes vs ImageNet ResNet-18:
    - conv1: 7x7 stride=2 -> 3x3 stride=1 padding=1
    - maxpool: removed (identity)
    - fc: removed (identity) so forward outputs 512-d features
    """

    backbone = resnet18(weights=None)

    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()

    return backbone


class ProjectorMLP(nn.Module):
    """MLP projector head: 512 -> 512 -> 128 with ReLU between."""

    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLR(nn.Module):
    """SimCLR encoder: backbone + projector.

    Forward returns L2-normalized embeddings (z) from the projector head.
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.backbone = build_resnet18_cifar10()
        self.projector = ProjectorMLP(in_dim=512, hidden_dim=512, out_dim=embedding_dim)

    def encode(self, x: torch.Tensor, *, normalize: bool = False) -> torch.Tensor:
        """Return backbone representations (before projector).

        For CIFAR ResNet-18 this is a 512-d vector after global average pooling.
        This is the representation used for kNN monitor and linear probing.
        """

        feats = self.backbone(x)  # (B, 512)
        if normalize:
            feats = F.normalize(feats, dim=1)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)  # (B, 512)
        z = self.projector(feats)  # (B, embedding_dim)
        z = F.normalize(z, dim=1)
        return z


class SupervisedResNet18CIFAR10(nn.Module):
    """Supervised baseline: backbone + linear classifier (512 -> 10)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = build_resnet18_cifar10()
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits
