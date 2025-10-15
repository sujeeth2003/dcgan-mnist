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

SEGMENTS = {0: "abcdef", 1: "bc", 2: "abdeg", 3: "abcdg", 4: "bcfg", 5: "acdfg", 6: "acdefg", 7: "abc", 8: "abcdefg", 9: "abcdfg"}
# segment endpoints on a 0..1 grid (x right, y down)
SEG_XY = {"a": ((.2, .1), (.8, .1)), "b": ((.8, .1), (.8, .5)), "c": ((.8, .5), (.8, .9)), "d": ((.2, .9), (.8, .9)),
          "e": ((.2, .5), (.2, .9)), "f": ((.2, .1), (.2, .5)), "g": ((.2, .5), (.8, .5))}


def _open(path):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def _find(directory, stem):
    for ext in ("", ".gz"):
        p = os.path.join(directory, stem + ext)
        if os.path.exists(p): return p
    raise FileNotFoundError(f"{stem}[.gz] not found in {directory}")


def load_mnist(directory):
    with _open(_find(directory, "train-images-idx3-ubyte")) as f:
        magic, n, r, c = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "not an IDX image file"
        x = np.frombuffer(f.read(), np.uint8).reshape(n, 1, r, c)
    with _open(_find(directory, "train-labels-idx1-ubyte")) as f:
        magic, n2 = struct.unpack(">II", f.read(8)); assert magic == 2049 and n2 == n
        y = np.frombuffer(f.read(), np.uint8)
    return torch.from_numpy(x.astype(np.float32) / 127.5 - 1.0), torch.from_numpy(y.astype(np.int64))


