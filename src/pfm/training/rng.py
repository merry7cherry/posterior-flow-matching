from __future__ import annotations

import random

import numpy as np
import torch


_RANK_SEED_STRIDE = 100_003
_NUMPY_MODULUS = 2**32 - 1


def rank_training_seed(base_seed: int, rank: int) -> int:
    base = int(base_seed)
    rank_value = int(rank)
    if rank_value < 0:
        raise ValueError("rank must be non-negative")
    return base + rank_value * _RANK_SEED_STRIDE


def seed_rank_training_rng(base_seed: int, rank: int) -> int:
    seed = rank_training_seed(base_seed, rank)
    random.seed(seed)
    np.random.seed(seed % _NUMPY_MODULUS)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
