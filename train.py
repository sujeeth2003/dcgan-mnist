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

