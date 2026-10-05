from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass
class DiagonalEstimate:
    value: torch.Tensor
    actual_vjp: int
    mean_squared_error_estimate: torch.Tensor | None


def estimate_variance_diagonal(
    gamma: torch.Tensor, x: torch.Tensor, probes: Iterable[torch.Tensor],
    count: int,
) -> DiagonalEstimate:
    """Unbiased Rademacher VJP diagonal, with streaming Monte Carlo variance.

    The caller supplies the complete effective variance graph and independent
    probes. Only first input derivatives at this state are evaluated.
    """
    if type(count) is not int or count < 1:
        raise ValueError("probe count must be positive")
    if not gamma.requires_grad:
        return DiagonalEstimate(torch.zeros_like(x), 0, torch.zeros((), device=x.device))
    mean = torch.zeros_like(x)
    m2 = torch.zeros_like(x) if count > 1 else None
    calls = 0
    for index, z in enumerate(probes):
        if index >= count:
            raise ValueError("too many derivative probes")
        vjp = torch.autograd.grad(
            gamma, x, grad_outputs=z, retain_graph=index + 1 < count,
            create_graph=False, allow_unused=True,
        )[0]
        calls += 1
        value = torch.zeros_like(x) if vjp is None else z * vjp
        delta = value - mean
        mean = mean + delta / (index + 1)
        if m2 is not None:
            m2 = m2 + delta * (value - mean)
    if calls != count:
        raise ValueError("too few derivative probes")
    error = None if m2 is None else m2.mean() / (count * (count - 1))
    return DiagonalEstimate(mean.detach(), calls, error)
