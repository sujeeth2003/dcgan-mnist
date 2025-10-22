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

