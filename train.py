"""Train a DCGAN and write sample grids.

    python train.py --data synthetic --epochs 20               # verified offline path
    python train.py --data mnist --mnist-dir data --epochs 25  # real MNIST (place the IDX files in ./data)

Prints per-epoch losses and the two diagnostics that tell you whether a GAN is healthy:
  D(x)    mean discriminator probability on real images        -> should hover around 0.5-0.8, not 1.0
  D(G(z)) mean discriminator probability on generated images    -> should hover around 0.2-0.5, not 0.0
If D(x) -> 1 and D(G(z)) -> 0 the discriminator has won and the generator has stopped learning.
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch import nn

from dcgan.data import load_mnist, synthetic_digits
from dcgan.models import Discriminator, Generator


def save_grid(x, path, nrow=8):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = (x.detach().cpu().clamp(-1, 1) + 1) / 2
    n = min(len(x), nrow * nrow); rows = (n + nrow - 1) // nrow
    fig, axes = plt.subplots(rows, nrow, figsize=(nrow * 0.9, rows * 0.9))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n: ax.imshow(x[i, 0], cmap="gray", vmin=0, vmax=1)
    fig.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=100); plt.close(fig)

