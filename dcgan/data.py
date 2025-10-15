"""Datasets of 28x28 grayscale digits, scaled to [-1, 1] (the range of the generator's tanh output).

  load_mnist(dir)      reads the four standard MNIST IDX files (gzip or raw) that you place in `dir`:
                       train-images-idx3-ubyte(.gz)  train-labels-idx1-ubyte(.gz)   from yann.lecun.com/exdb/mnist or any mirror.
                       No download happens in this repo, and no torchvision is needed.
  synthetic_digits(n)  procedurally drawn 7-segment digits with random stroke width, shift, slant and noise. It is NOT MNIST;
                       it exists so the full pipeline can be verified anywhere, offline, in minutes on a CPU.
"""
import gzip
import os
import struct

import numpy as np
import torch

