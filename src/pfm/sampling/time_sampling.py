from __future__ import annotations

import torch

from pfm.core.config import ExperimentConfig


def uniform_timestep_sample(
    config: ExperimentConfig,
    num_samples: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return (
        torch.rand((num_samples,), device=device, dtype=torch.float32, generator=generator)
        * (1.0 - 2.0 * config.eps_t)
        + config.eps_t
    )


def logit_normal_timestep_sample(
    P_mean: float,
    P_std: float,
    num_samples: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    rnd_normal = torch.randn((num_samples,), device=device, dtype=torch.float32, generator=generator)
    time = torch.sigmoid(rnd_normal * P_std + P_mean)
    time = torch.clamp(time, min=0.0, max=1.0)
    return time


def sample_train_timesteps(
    config: ExperimentConfig,
    num_samples: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    mode = config.time_sampling
    if mode == "uniform":
        t = uniform_timestep_sample(config, num_samples, device, generator=generator)
    elif mode == "logit_normal":
        t = logit_normal_timestep_sample(
            P_mean=config.logit_normal_p_mean,
            P_std=config.logit_normal_p_std,
            num_samples=num_samples,
            device=device,
            generator=generator,
        )
        t = torch.clamp(t, min=config.eps_t, max=1.0 - config.eps_t)
    else:
        raise ValueError(f"Unknown time_sampling {mode!r}")
    return t.view(num_samples, 1, 1, 1)
