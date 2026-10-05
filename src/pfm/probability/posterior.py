from __future__ import annotations

from dataclasses import dataclass

import torch

from pfm.core.config import ExperimentConfig
from pfm.core.variant_metadata import DELTA_VARIANTS, variant_capabilities
from pfm.probability.diagonal_gaussian import variance_from_logvar
from pfm.probability.logvar import bound_normalized_logvar
from pfm.probability.targets import time_like


@dataclass
class PosteriorPrediction:
    velocity: torch.Tensor
    endpoint_mean: torch.Tensor
    velocity_var: torch.Tensor | None
    endpoint_var: torch.Tensor | None
    effective_logvar: torch.Tensor | None


def posterior_from_outputs(
    outputs: dict[str, torch.Tensor], variant: str, x: torch.Tensor,
    t: torch.Tensor, config: ExperimentConfig,
) -> PosteriorPrediction:
    """Interpret both PFM heads in velocity units, preserving input AD."""
    xt = x.float()
    time = time_like(t.float(), xt)
    if variant in DELTA_VARIANTS or variant == "velocity_fm":
        velocity = outputs["delta" if variant in DELTA_VARIANTS else "velocity"].float()
        endpoint = xt + (1.0 - time) * velocity
    elif variant in {"pfm_full", "endpoint_fm"}:
        endpoint = outputs["mu" if variant == "pfm_full" else "x1_hat"].float()
        velocity = (endpoint - xt) / (1.0 - time)
    else:
        raise ValueError(f"Unknown variant {variant!r}")
    logvar = gamma = endpoint_var = None
    if variant_capabilities(variant).predicts_logvar:
        logvar = bound_normalized_logvar(outputs["logvar"], config)
        gamma = variance_from_logvar(logvar)
        endpoint_var = (1.0 - time).square() * gamma
    return PosteriorPrediction(velocity, endpoint, gamma, endpoint_var, logvar)


def score_from_velocity(
    x: torch.Tensor, t: torch.Tensor, velocity: torch.Tensor, noise_scale: float,
) -> torch.Tensor:
    """Gaussian-source identity, with the checkpoint's source scale."""
    time = time_like(t.float(), x)
    return (time * velocity.float() - x.float()) / (float(noise_scale) ** 2 * (1.0 - time))


def sde_coefficients(
    velocity: torch.Tensor, gamma: torch.Tensor, score: torch.Tensor,
    divergence: torch.Tensor, diffusion_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rate = float(diffusion_scale) * gamma
    drift = velocity + 0.5 * float(diffusion_scale) * (divergence + gamma * score)
    return drift, rate
