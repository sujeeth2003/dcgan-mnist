"""Quantitative check of a trained generator (a picture grid alone can fool you):

  * train a small classifier on REAL images, then classify GENERATED ones:
      - confidence  : mean max-softmax  (blurry/garbage images give low confidence)
      - class coverage / entropy : are all 10 digits produced roughly equally? (mode collapse shows as a few dominant classes)
  * pixel statistics vs real data.

    python evaluate.py --data synthetic --epochs 20
"""
import argparse
import math

import torch
import torch.nn.functional as F
from torch import nn

from dcgan.data import load_mnist, synthetic_digits
from train import train


def make_classifier():
    return nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                         nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64 * 7 * 7, 10))


def fit_classifier(x, y, epochs=4):
    torch.manual_seed(0)
    clf, opt = make_classifier(), None
    opt = torch.optim.Adam(clf.parameters(), 1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 128):
            idx = perm[i:i + 128]; opt.zero_grad(); F.cross_entropy(clf(x[idx]), y[idx]).backward(); opt.step()
    return clf.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "mnist"], default="synthetic")
    ap.add_argument("--mnist-dir", default="data"); ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--n", type=int, default=6000)
    a = ap.parse_args()
    x, y = load_mnist(a.mnist_dir) if a.data == "mnist" else synthetic_digits(a.n)
    xt, yt = (x[: int(.85 * len(x))], y[: int(.85 * len(x))]), (x[int(.85 * len(x)):], y[int(.85 * len(x)):])
    clf = fit_classifier(*xt)
    with torch.no_grad(): acc = (clf(yt[0]).argmax(1) == yt[1]).float().mean().item()
    print(f"reference classifier on held-out REAL images: {acc:.1%} accuracy")
    G, _, hist = train(xt[0], a.epochs, log=lambda s: None)
    with torch.no_grad():
        fake = G(torch.randn(2000, 100, 1, 1)); p = F.softmax(clf(fake), 1)
    conf = p.max(1).values.mean().item(); counts = torch.bincount(p.argmax(1), minlength=10).float(); q = counts / counts.sum()
    ent = -(q[q > 0] * q[q > 0].log()).sum().item() / math.log(10)
    print(f"generated images ({a.epochs} epochs): mean classifier confidence {conf:.2f}; classes produced: {int((counts > 0).sum())}/10; "
          f"class entropy {ent:.2f} (1.00 = perfectly balanced)")
    print("class histogram:", counts.int().tolist())
    print(f"pixel mean real {xt[0].mean():.2f} / fake {fake.mean():.2f};  std real {xt[0].std():.2f} / fake {fake.std():.2f}")
    print(f"final D(x) {hist[-1][2]:.2f}  D(G(z)) {hist[-1][3]:.2f}")


if __name__ == "__main__":
    main()
