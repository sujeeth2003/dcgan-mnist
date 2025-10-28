import gzip
import os
import struct
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dcgan.data import load_mnist, synthetic_digits  # noqa: E402
from dcgan.models import Discriminator, Generator  # noqa: E402
from train import train  # noqa: E402


class DCGANTests(unittest.TestCase):
    def test_shapes_and_ranges(self):
        g, d = Generator(), Discriminator()
        x = g(torch.randn(5, 100, 1, 1))
        self.assertEqual(tuple(x.shape), (5, 1, 28, 28)); self.assertTrue(x.min() >= -1 and x.max() <= 1)
        self.assertEqual(tuple(d(x).shape), (5,))

    def test_architecture_follows_the_paper(self):
        g, d = Generator(), Discriminator()
        for net in (g, d):
            self.assertFalse(any(isinstance(m, (torch.nn.MaxPool2d, torch.nn.Linear)) for m in net.modules()))   # no pooling / FC layers
        w = g.net[0].weight
        self.assertAlmostEqual(float(w.std()), 0.02, delta=0.005)                                              # N(0, 0.02) init

    def test_synthetic_digits_are_distinguishable_classes(self):
        x, y = synthetic_digits(600, seed=1)
        self.assertEqual(tuple(x.shape), (600, 1, 28, 28)); self.assertTrue(x.min() >= -1 and x.max() <= 1)
        m = torch.stack([x[y == k].mean(0) for k in range(10)])
        self.assertGreater(float((m[1] - m[8]).abs().mean()), 0.05)          # '1' and '8' look different on average

    def test_mnist_idx_reader(self):
        d = tempfile.mkdtemp(); imgs = (np.arange(3 * 28 * 28) % 256).astype(np.uint8)
        with gzip.open(os.path.join(d, "train-images-idx3-ubyte.gz"), "wb") as f: f.write(struct.pack(">IIII", 2051, 3, 28, 28) + imgs.tobytes())
        with gzip.open(os.path.join(d, "train-labels-idx1-ubyte.gz"), "wb") as f: f.write(struct.pack(">II", 2049, 3) + bytes([7, 2, 1]))
        x, y = load_mnist(d)
        self.assertEqual(tuple(x.shape), (3, 1, 28, 28)); self.assertEqual(y.tolist(), [7, 2, 1])
        self.assertAlmostEqual(float(x.max()), 1.0, places=1)

    def test_training_moves_the_generator_toward_the_data(self):
        x, _ = synthetic_digits(768, seed=2)
        torch.manual_seed(0); before = Generator()(torch.randn(256, 100, 1, 1)).mean().item()
        G, D, hist = train(x, epochs=6, log=lambda s: None)
        with torch.no_grad(): after = G(torch.randn(256, 100, 1, 1)).mean().item()
        real = x.mean().item()
        self.assertLess(abs(after - real), abs(before - real))               # background level (mostly -1) learned
        self.assertTrue(all(np.isfinite(h).all() for h in hist))


if __name__ == "__main__":
    unittest.main()
