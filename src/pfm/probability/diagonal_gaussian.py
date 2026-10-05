from __future__ import annotations

import torch


def variance_from_logvar(logvar: torch.Tensor) -> torch.Tensor:
    return torch.exp(logvar.to(dtype=torch.float32))


def squared_residual(residual: torch.Tensor) -> torch.Tensor:
    return residual.to(dtype=torch.float32).square()


def constant_free_nll_per_component(
    residual: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    logvar_fp32 = logvar.to(dtype=torch.float32)
    return 0.5 * (torch.exp(-logvar_fp32) * squared_residual(residual) + logvar_fp32)


def constant_free_nll_per_image(
    residual: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    return constant_free_nll_per_component(residual, logvar).flatten(1).mean(dim=1)


def constant_free_nll(
    residual: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    return constant_free_nll_per_image(residual, logvar).mean()
