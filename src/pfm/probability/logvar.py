from __future__ import annotations

import math

import torch

from pfm.core.config import ExperimentConfig


_BOUNDARY_PROBABILITY_EPS = 1e-4


def bound_normalized_logvar(raw_logvar: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
    low, high = _logvar_bounds(config)
    raw_logvar_fp32 = raw_logvar.to(dtype=torch.float32)
    return low + (high - low) * torch.sigmoid(raw_logvar_fp32)


def inverse_smooth_logvar_bound(value: float, config: ExperimentConfig) -> float:
    low, high = _logvar_bounds(config)
    value = float(value)
    if not low <= value <= high:
        raise ValueError("logvar_init must lie between logvar_min and logvar_max")
    if value == low:
        probability = _BOUNDARY_PROBABILITY_EPS
    elif value == high:
        probability = 1.0 - _BOUNDARY_PROBABILITY_EPS
    else:
        probability = (value - low) / (high - low)
    return math.log(probability / (1.0 - probability))


def _logvar_bounds(config: ExperimentConfig) -> tuple[float, float]:
    low = float(config.logvar_min)
    high = float(config.logvar_max)
    if not low < high:
        raise ValueError("logvar_min must be smaller than logvar_max")
    return low, high
