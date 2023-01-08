"""
Utiliy functions for the project. Contains constants.
"""

import os
import torch

RANDOM_SEED = 69
device = "cuda" if torch.cuda.is_available() else "cpu"