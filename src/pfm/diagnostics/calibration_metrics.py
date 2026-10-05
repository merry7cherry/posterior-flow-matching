from __future__ import annotations

import torch

from pfm.core.config import ExperimentConfig
from pfm.core.variant_metadata import requires_learned_variance, method_profile_name
from pfm.diagnostics.posterior_residuals import (
    collect_posterior_residual_statistics,
    posterior_generator,
    posterior_histogram_edges,
    summarize_time_calibration,
    summarize_variance_calibration,
)
from pfm.sampling.time_sampling import sample_train_timesteps


@torch.no_grad()
def evaluate_pfm_calibration(
    model: torch.nn.Module,
    variant: str,
    config: ExperimentConfig,
    data_loader,
    device: torch.device,
    num_samples: int,
    num_bins: int = 15,
    seed: int = 321,
) -> dict[str, object]:
    requires_learned_variance(variant, feature="posterior_calibration")
    method_profile = method_profile_name(variant)
    time_generator = posterior_generator(device, seed, "random_time", method_profile, "time")
    noise_generator = posterior_generator(device, seed, "random_time", method_profile, "noise")

    def training_time_sampler(
        sampler_config: ExperimentConfig,
        batch_size: int,
        sampler_device: torch.device,
        sampler_variant: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        del dtype
        return sample_train_timesteps(
            sampler_config,
            batch_size,
            device=sampler_device,
            generator=time_generator,
        )

    residual_stats = collect_posterior_residual_statistics(
        model=model,
        variant=variant,
        config=config,
        data_loader=data_loader,
        device=device,
        num_samples=int(num_samples),
        num_bins=int(num_bins),
        t_sampler=training_time_sampler,
        include_time_calibration=True,
        noise_generator=noise_generator,
        histogram_edges=posterior_histogram_edges(
            config,
            num_bins=num_bins,
            random_time_endpoint=True,
        ),
    )
    return {
        "available": True,
        "variant": variant,
        "method_profile": method_profile,
        "rng": {
            "base_seed": int(seed),
            "suite": "random_time",
            "profile": method_profile,
        },
        **residual_stats,
    }
