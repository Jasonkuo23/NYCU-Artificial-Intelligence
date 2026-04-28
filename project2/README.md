# SimCLR Baseline (CIFAR-10, PyTorch)

This folder contains a clean, minimal SimCLR-style self-supervised baseline.

## Run

```bash
pip install -r requirements.txt
python3 train_simclr.py --data-dir ./data --epochs 200 --batch-size 512
```

### SSL pretraining (SimCLR) + kNN monitor

Runs NT-Xent training and prints per-epoch loss. Every 5 epochs (default), it also prints kNN monitor accuracy
using the backbone representations (512-d) on CIFAR-10.

```bash
python3 train_simclr.py --data-dir ./data --epochs 200 --batch-size 512 \
	--knn-every 5 --knn-k 20 --save-path ./checkpoints/simclr_last.pt
```

### Linear probing (required evaluation)

Freeze the encoder and train only a linear classifier (512 -> 10).

```bash
python3 linear_probe.py --data-dir ./data --checkpoint ./checkpoints/simclr_last.pt \
	--epochs 100 --batch-size 512
```

"Random frozen" lower bound (no SSL checkpoint):

```bash
python3 linear_probe.py --data-dir ./data --epochs 100 --batch-size 512
```

### Supervised baseline (required comparison)

Train the same CIFAR-modified ResNet-18 end-to-end with a classification head.

```bash
python3 train_supervised.py --data-dir ./data --epochs 200 --batch-size 512
```

## Files

- simclr/dataset.py: CIFAR-10 + SimCLR augmentations producing two views
- simclr/model.py: ResNet-18 (CIFAR) + projector MLP
- simclr/loss.py: NT-Xent loss (matrix-based cosine similarity)
- simclr/eval.py: feature extraction + kNN monitor
- train_simclr.py: SSL training loop (Adam) + kNN monitor + checkpoint
- linear_probe.py: linear probing on frozen encoder
- train_supervised.py: supervised-from-scratch baseline
