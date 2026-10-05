from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from pfm.core.config import ExperimentConfig
from pfm.core.variant_metadata import DELTA_VARIANTS, SUPPORTED_VARIANTS
from pfm.probability.diagonal_gaussian import (
    constant_free_nll,
    squared_residual,
    variance_from_logvar,
)
from pfm.probability.logvar import bound_normalized_logvar
from pfm.probability.targets import direct_velocity_target, endpoint_to_velocity, velocity_target


TensorDict = Dict[str, Optional[torch.Tensor]]


def compute_loss(
    variant: str,
    outputs: TensorDict,
    x0: torch.Tensor,
    x1: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
    config: ExperimentConfig,
) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}")

    if variant == "velocity_fm":
        residual = direct_velocity_target(x0, x1) - _require(outputs, "velocity")
        residual_mse = residual.pow(2).mean()
        loss = 0.5 * residual_mse
        return loss, {
            "loss": loss.detach(),
            "residual_mse": residual_mse.detach(),
        }

    if variant in DELTA_VARIANTS:
        raw_logvar = _require(outputs, "logvar")
        logvar = bound_normalized_logvar(raw_logvar, config)
        residual = direct_velocity_target(x0, x1) - _require(outputs, "delta")
        loss = constant_free_nll(residual, logvar)
        stats = {
            "loss": loss.detach(),
            "residual_mse": squared_residual(residual).mean().detach(),
            "velocity_variance_mean": variance_from_logvar(logvar).mean().detach(),
            "logvar_mean": logvar.mean().detach(),
            "logvar_raw_mean": raw_logvar.float().mean().detach(),
        }
        return loss, stats

    v_target = velocity_target(x1, xt, t)
    if variant == "pfm_full":
        mu = _require(outputs, "mu")
        raw_logvar = _require(outputs, "logvar")
        logvar = bound_normalized_logvar(raw_logvar, config)
        residual = v_target - endpoint_to_velocity(mu, xt, t)
        loss = constant_free_nll(residual, logvar)
        stats = {
            "loss": loss.detach(),
            "residual_mse": squared_residual(residual).mean().detach(),
            "velocity_variance_mean": variance_from_logvar(logvar).mean().detach(),
            "logvar_mean": logvar.mean().detach(),
            "logvar_raw_mean": raw_logvar.float().mean().detach(),
        }
        return loss, stats

    if variant == "endpoint_fm":
        residual = v_target - endpoint_to_velocity(_require(outputs, "x1_hat"), xt, t)
    else:
        raise ValueError(f"Unhandled variant {variant!r}")
    residual_mse = residual.pow(2).mean()
    loss = 0.5 * residual_mse
    return loss, {
        "loss": loss.detach(),
        "residual_mse": residual_mse.detach(),
    }


def _require(outputs: TensorDict, key: str) -> torch.Tensor:
    value = outputs.get(key)
    if value is None:
        raise KeyError(f"Expected output {key!r}")
    return value
