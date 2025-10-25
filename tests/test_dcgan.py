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

