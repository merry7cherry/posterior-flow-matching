from __future__ import annotations

import torch


def direct_velocity_target(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    return x1 - x0


def velocity_target(x1: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (x1 - xt) / (1.0 - time_like(t, x1))


def endpoint_to_velocity(endpoint: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (endpoint - xt) / (1.0 - time_like(t, endpoint))


def time_like(t: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if t.ndim == 1:
        return t.view(t.shape[0], *([1] * (reference.ndim - 1)))
    while t.ndim < reference.ndim:
        t = t.unsqueeze(-1)
    return t
