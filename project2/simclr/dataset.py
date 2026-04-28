from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# CIFAR-10 normalization (commonly used for training ResNets on CIFAR-10)
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class TwoCropsTransform:
    """Create two augmented views of the same image.

    In SimCLR, each sample produces two *different* stochastic augmentations.
    The contrastive loss will treat these as a positive pair.
    """

    def __init__(self, base_transform: transforms.Compose):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k


class TwoViewCIFAR10(datasets.CIFAR10):
    """CIFAR-10 that returns exactly two augmented views.

    torchvision's CIFAR-10 returns (image, label). For SimCLR pretraining we
    typically ignore labels, so this dataset returns (view1, view2).
    """

    def __getitem__(self, index: int):
        (views, _target) = super().__getitem__(index)
        # With TwoCropsTransform, `views` is a tuple: (view1, view2)
        return views


@dataclass
class SimCLRAugmentationConfig:
    image_size: int = 32
    # RandomResizedCrop scale range used in SimCLR
    crop_scale: tuple[float, float] = (0.2, 1.0)
    # Color jitter strength (SimCLR uses 0.8 * [0.4, 0.4, 0.4, 0.1])
    jitter_strength: float = 0.5
    jitter_prob: float = 0.8
    grayscale_prob: float = 0.2
    horizontal_flip_prob: float = 0.5


def build_simclr_transform(cfg: SimCLRAugmentationConfig | None = None) -> transforms.Compose:
    """SimCLR-style data augmentation pipeline for CIFAR-10."""

    cfg = cfg or SimCLRAugmentationConfig()

    color_jitter = transforms.ColorJitter(
        0.8 * cfg.jitter_strength,
        0.8 * cfg.jitter_strength,
        0.8 * cfg.jitter_strength,
        0.2 * cfg.jitter_strength,
    )

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(cfg.image_size, scale=cfg.crop_scale),
            transforms.RandomHorizontalFlip(p=cfg.horizontal_flip_prob),
            transforms.RandomApply([color_jitter], p=cfg.jitter_prob),
            transforms.RandomGrayscale(p=cfg.grayscale_prob),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
        ]
    )


def build_cifar10_simclr_dataloader(
    *,
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """CIFAR-10 dataloader yielding (view1, view2) for self-supervised training."""

    base_transform = build_simclr_transform()
    two_view_transform = TwoCropsTransform(base_transform)

    # We only need the training split for self-supervised pretraining.
    dataset = TwoViewCIFAR10(root=data_dir, train=True, download=True, transform=two_view_transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # important: keeps batch size constant for contrastive pairing
    )


def build_cifar10_eval_transform() -> transforms.Compose:
    """Deterministic transform for evaluation (no augmentation)."""

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
        ]
    )


def build_cifar10_eval_dataloaders(
    *,
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Train/test loaders for evaluation (used by kNN monitor & linear probing)."""

    t = build_cifar10_eval_transform()
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=t)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=t)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def build_cifar10_supervised_dataloaders(
    *,
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Standard supervised CIFAR-10 dataloaders."""

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
        ]
    )
    test_transform = build_cifar10_eval_transform()

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader
