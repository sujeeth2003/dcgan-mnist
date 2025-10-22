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


def train(x, epochs=20, batch=128, nz=100, lr=2e-4, seed=0, out_dir=None, log=print, device="cpu"):
    torch.manual_seed(seed)
    G, D = Generator(nz).to(device), Discriminator().to(device)
    oG = torch.optim.Adam(G.parameters(), lr, betas=(0.5, 0.999)); oD = torch.optim.Adam(D.parameters(), lr, betas=(0.5, 0.999))
    fixed = torch.randn(64, nz, 1, 1, device=device)
    hist = []
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(x)); ld = lg = dx = dgz = 0.0; nb = 0
        for i in range(0, len(x) - batch + 1, batch):
            real = x[perm[i:i + batch]].to(device)
            z = torch.randn(batch, nz, 1, 1, device=device); fake = G(z)
            # --- discriminator: real -> 1 (smoothed to 0.9), fake -> 0
            oD.zero_grad()
            lr_ = F.binary_cross_entropy_with_logits(D(real), torch.full((batch,), 0.9, device=device))
            lf_ = F.binary_cross_entropy_with_logits(D(fake.detach()), torch.zeros(batch, device=device))
            (lr_ + lf_).backward(); oD.step()
            # --- generator: make D say 1 on fakes (the non-saturating loss: -log D(G(z)) instead of log(1 - D(G(z))))
            oG.zero_grad()
            out = D(fake); lg_ = F.binary_cross_entropy_with_logits(out, torch.ones(batch, device=device)); lg_.backward(); oG.step()
            ld += (lr_ + lf_).item(); lg += lg_.item(); nb += 1
            with torch.no_grad():
                dx += torch.sigmoid(D(real)).mean().item(); dgz += torch.sigmoid(out).mean().item()
        hist.append((ld / nb, lg / nb, dx / nb, dgz / nb))
        log(f"epoch {ep:3d}  loss D {ld / nb:.3f}  G {lg / nb:.3f}   D(x) {dx / nb:.2f}  D(G(z)) {dgz / nb:.2f}")
        if out_dir and (ep in (1, 5, 10) or ep % 10 == 0 or ep == epochs):
            G.eval(); save_grid(G(fixed), os.path.join(out_dir, f"epoch_{ep:03d}.png")); G.train()
    G.eval()
    return G, D, hist

