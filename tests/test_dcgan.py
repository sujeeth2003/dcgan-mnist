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

