from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import math
import time

import torch

from pfm.core.config import ExperimentConfig
from pfm.core.sampling_config import SamplingConfig
from pfm.core.variant_metadata import validate_base_model_config, requires_learned_variance
from pfm.probability.posterior import posterior_from_outputs, score_from_velocity, sde_coefficients
from pfm.sampling.derivatives import estimate_variance_diagonal
from pfm.sampling.randomness import indexed_step_random


@dataclass
class SampleResult:
    samples: torch.Tensor
    endpoint_var: torch.Tensor | None
    velocity_var: torch.Tensor | None
    variance_available: bool
    sampling_steps: int
    actual_nfe: int
    uncertainty_source_step: int | None
    uncertainty_source_t: float | None
    uncertainty_source_state: str | None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    actual_vjp: int = 0
    diffusion_rate: torch.Tensor | None = None
    noise_covariance: torch.Tensor | None = None
    numerical_summary: dict[str, float | int | None] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


@dataclass
class VelocityPrediction:
    velocity: torch.Tensor
    endpoint_var: torch.Tensor | None
    velocity_var: torch.Tensor | None
    outputs: dict[str, torch.Tensor]
    endpoint_mean: torch.Tensor | None = None
    actual_nfe: int = 1


def predict_velocity_and_uncertainty(
    model: torch.nn.Module, variant: str, x: torch.Tensor, t: torch.Tensor,
    labels: torch.Tensor, config: ExperimentConfig, *, precision: str | None = None,
) -> VelocityPrediction:
    mode = precision or ('bf16' if config.mixed_precision == 'bf16' else 'fp32')
    context = (
        torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=mode == 'bf16')
        if x.device.type == 'cuda' else nullcontext()
    )
    with context:
        outputs = model(x, t.reshape(-1), labels)
    posterior = posterior_from_outputs(outputs, variant, x, t, config)
    return VelocityPrediction(
        posterior.velocity, posterior.endpoint_var, posterior.velocity_var,
        outputs, posterior.endpoint_mean,
    )


def _capture_step(capture_t: float | None, sampling: SamplingConfig) -> int | None:
    if capture_t is None:
        return None
    if isinstance(capture_t, bool) or not isinstance(capture_t, (int, float)):
        raise TypeError('uncertainty_capture_t must be a number or None')
    if not math.isfinite(capture_t) or not 0 <= capture_t < sampling.terminal_time:
        raise ValueError('uncertainty_capture_t must be before terminal_time and in [0, 1)')
    step = round(capture_t * sampling.steps / sampling.terminal_time)
    if step >= sampling.steps or abs(step * sampling.terminal_time / sampling.steps - capture_t) > 1e-12:
        raise ValueError('uncertainty_capture_t must be exactly representable by the sampling grid')
    return step


@torch.no_grad()
def sample(
    model: torch.nn.Module, variant: str, labels: torch.Tensor,
    config: ExperimentConfig, sampling: SamplingConfig, device: torch.device,
    *, initial_noise: torch.Tensor | None = None,
    sample_indices: torch.Tensor | None = None, seed: int = 0,
    uncertainty_capture_t: float | None = None,
    generator: torch.Generator | None = None,
) -> SampleResult:
    """Shared-grid Euler / corrected Ito Euler--Maruyama generation.

    Only the current SDE state's effective variance retains an input graph.
    Derivative probes are independent of Brownian increments and initialization.
    """
    if not isinstance(sampling, SamplingConfig):
        raise TypeError('sampling must be a SamplingConfig')
    validate_base_model_config(variant, config)
    if sampling.solver == 'sde':
        requires_learned_variance(variant, feature='SDE sampling')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('sampling seed must be an integer in [0, 2**32)')
    if torch.is_inference_mode_enabled() and sampling.solver == 'sde' and sampling.diffusion_scale > 0:
        raise RuntimeError('SDE input derivatives require no_grad/enable_grad, not inference_mode')
    capture_step = _capture_step(uncertainty_capture_t, sampling)
    model.eval()
    labels = labels.to(device)
    shape = (labels.shape[0], config.image_channels, config.image_size, config.image_size)
    if initial_noise is None:
        # The generation service supplies its existing indexed initialization.
        if generator is None:
            generator = torch.Generator(device=device).manual_seed(seed)
        x = float(config.noise_scale) * torch.randn(shape, device=device, generator=generator)
    else:
        if tuple(initial_noise.shape) != shape:
            raise ValueError(f'initial_noise must have shape {shape}, got {tuple(initial_noise.shape)}')
        x = initial_noise.detach().to(device=device, dtype=torch.float32).clone()
    indices = torch.arange(shape[0]) if sample_indices is None else sample_indices
    if indices.ndim != 1 or indices.numel() != shape[0] or indices.dtype != torch.long:
        raise ValueError('sample_indices must be an int64 vector matching the batch')
    if torch.any(indices < 0):
        raise ValueError('sample_indices must be nonnegative')
    endpoint_var = velocity_var = rate_map = noise_map = None
    source_step = source_t = None
    actual_nfe = actual_vjp = 0
    stats = torch.zeros(5, device=device, dtype=torch.float64)
    error_available = True
    finite = torch.isfinite(x).all()
    parameter_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    active_sde = sampling.solver == 'sde' and sampling.diffusion_scale > 0
    started = time.perf_counter()
    h = sampling.terminal_time / sampling.steps
    try:
        if active_sde:
            for parameter, _ in parameter_flags:
                parameter.requires_grad_(False)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for step in range(sampling.steps):
            t_scalar = step * sampling.terminal_time / sampling.steps
            t = torch.full((shape[0], 1, 1, 1), t_scalar, device=device, dtype=torch.float32)
            lam = sampling.lambda_at(t_scalar) if active_sde else 0.0
            if lam > 0:
                with torch.enable_grad():
                    current = x.detach().requires_grad_(True)
                    pred = predict_velocity_and_uncertainty(model, variant, current, t, labels, config, precision=sampling.precision)
                    gamma = pred.velocity_var
                    if gamma is None:
                        raise ValueError('SDE requires velocity variance')
                    probes = (
                        indexed_step_random(indices, shape[1:], seed=seed, step=step,
                                            stream='derivative', probe=probe, device=device)
                        for probe in range(sampling.derivative_probes)
                    )
                    estimate = estimate_variance_diagonal(gamma, current, probes, sampling.derivative_probes)
                gamma = gamma.detach()
                velocity = pred.velocity.detach()
                score = score_from_velocity(x, t, velocity, config.noise_scale)
                drift, rate = sde_coefficients(velocity, gamma, score, estimate.value, lam)
                xi = indexed_step_random(indices, shape[1:], seed=seed, step=step, stream='brownian', device=device)
                next_x = x + h * drift + (h * rate).sqrt() * xi
                actual_vjp += estimate.actual_vjp
                stats[0] += score.square().mean().double()
                stats[1] += estimate.value.square().mean().double()
                stats[2] += (drift - velocity).square().mean().double()
                stats[3] += (h * rate).mean().double()
                if estimate.mean_squared_error_estimate is None:
                    error_available = False
                else:
                    stats[4] += (h * lam / 2) ** 2 * estimate.mean_squared_error_estimate.double()
                finite &= torch.isfinite(score).all() & torch.isfinite(estimate.value).all()
            else:
                pred = predict_velocity_and_uncertainty(model, variant, x, t, labels, config, precision=sampling.precision)
                next_x = x + h * pred.velocity
                rate = None
            actual_nfe += pred.actual_nfe
            if (capture_step is None or step == capture_step) and pred.velocity_var is not None:
                velocity_var = pred.velocity_var.detach()
                endpoint_var = pred.endpoint_var.detach()
                source_step, source_t = step, t_scalar
                if sampling.solver == 'sde':
                    rate_map = torch.zeros_like(velocity_var) if rate is None else rate.detach()
                    noise_map = h * rate_map
            finite &= torch.isfinite(next_x).all()
            if pred.velocity_var is not None:
                finite &= torch.isfinite(pred.velocity_var).all() & (pred.velocity_var > 0).all()
            x = next_x.detach()
            del pred
        if sampling.terminal_readout == 'endpoint_mean':
            t = torch.full((shape[0], 1, 1, 1), sampling.terminal_time, device=device, dtype=torch.float32)
            pred = predict_velocity_and_uncertainty(model, variant, x, t, labels, config, precision=sampling.precision)
            x = pred.endpoint_mean.detach()
            actual_nfe += pred.actual_nfe
            finite &= torch.isfinite(x).all()
    finally:
        if active_sde:
            for parameter, flag in parameter_flags:
                parameter.requires_grad_(flag)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if not bool(finite & torch.isfinite(stats).all()):
        raise FloatingPointError('nonfinite state or invalid SDE coefficient during sampling')
    moments = (stats / sampling.steps).cpu().tolist()
    numerical_summary = dict(zip(
        ('score_mean_square', 'divergence_mean_square', 'correction_mean_square', 'expected_brownian_increment_mean_square', 'derivative_update_mse_estimate'),
        moments,
    ))
    numerical_summary['nonfinite_count'] = 0
    if not error_available:
        numerical_summary['derivative_update_mse_estimate'] = None
    return SampleResult(
        x, endpoint_var, velocity_var, velocity_var is not None,
        sampling.steps, actual_nfe, source_step, source_t,
        'pre_update' if source_step is not None else None,
        sampling, actual_vjp, rate_map, noise_map, numerical_summary, elapsed,
    )
