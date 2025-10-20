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


