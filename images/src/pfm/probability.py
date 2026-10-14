"""PFM likelihood and velocity/endpoint posterior fields (FP32)."""
import math
from dataclasses import dataclass
import torch


def time_like(t, x):
    return t.float().reshape(-1, *([1] * (x.ndim - 1)))


def bound_logvar(raw, config):
    return config.logvar_min + (config.logvar_max - config.logvar_min) * raw.float().sigmoid()


def inverse_smooth_logvar_bound(value, config):
    p = (value - config.logvar_min) / (config.logvar_max - config.logvar_min)
    if not 0 <= p <= 1:
        raise ValueError("logvar_init must lie within the configured bounds")
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


@dataclass
class Posterior:
    velocity: torch.Tensor
    endpoint_mean: torch.Tensor
    logvar: torch.Tensor
    velocity_var: torch.Tensor
    endpoint_var: torch.Tensor


def posterior(outputs, x, t, config):
    x, t = x.float(), time_like(t, x)
    mean = outputs["mean"].float()
    if config.mean_parameterization == "velocity":
        velocity, endpoint = mean, x + (1 - t) * mean
    else:
        endpoint, velocity = mean, (mean - x) / (1 - t)
    logvar = bound_logvar(outputs["logvar"], config)
    gamma = logvar.exp()
    return Posterior(velocity, endpoint, logvar, gamma, (1 - t).square() * gamma)


def pfm_loss(outputs, x0, x1, xt, t, config):
    prediction = posterior(outputs, xt, t, config)
    # Endpoint prediction uses the same induced-velocity target as the native implementation.
    target = x1.float() - x0.float() if config.mean_parameterization == "velocity" else (x1.float() - xt.float()) / (1 - time_like(t, xt))
    residual = target - prediction.velocity
    return (0.5 * (residual.square() * (-prediction.logvar).exp() + prediction.logvar)).flatten(1).mean(1).mean()


def score_from_velocity(x, t, velocity, noise_scale):
    t = time_like(t, x)
    return (t * velocity.float() - x.float()) / (noise_scale ** 2 * (1 - t))


def sde_coefficients(velocity, gamma, score, divergence, lam):
    return velocity + 0.5 * lam * (divergence + gamma * score), lam * gamma
