"""DCGAN (Radford, Metz, Chintala 2016) for 28x28 grayscale images.

The paper's recipe: no pooling or fully-connected layers (strided / transposed convolutions instead), BatchNorm in both networks
(not on the generator output or discriminator input), ReLU in G with tanh out, LeakyReLU(0.2) in D, weights ~ N(0, 0.02).
"""
import torch
from torch import nn


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02); nn.init.zeros_(m.bias)


class Generator(nn.Module):
    """z (B, nz, 1, 1) -> image (B, 1, 28, 28) in [-1, 1].   1x1 -> 7x7 -> 14x14 -> 28x28"""

    def __init__(self, nz=100, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 7, 1, 0, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True),         # 7x7
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),    # 14x14
            nn.ConvTranspose2d(ngf * 2, 1, 4, 2, 1, bias=False), nn.Tanh())                                       # 28x28
        self.apply(init_weights)

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """image (B, 1, 28, 28) -> logit (B,).  28 -> 14 -> 7 -> 1"""

    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),                                      # 14x14 (no BN on input layer)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),       # 7x7
            nn.Conv2d(ndf * 2, 1, 7, 1, 0, bias=False))                                                           # 1x1 logit
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x).view(-1)
