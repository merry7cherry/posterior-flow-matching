"""PFM ODE and corrected Ito SDE; no graph is retained across integration steps."""

import torch
from .pfm_probability import (
    center,
    covariance_apply,
    noise_from_standard,
    projected_divergence,
    score_from_velocity,
    marginal_variance,
)


def sample(
    predict,
    x,
    batch,
    *,
    steps=50,
    terminal_time=0.99,
    mode="ode",
    diffusion_scale=0.3,
    probes=2,
    source_scale=1.0,
    seed=42,
    save_trajectory=False,
):
    if mode not in {"ode", "sde"} or steps < 1 or not 0 < terminal_time < 1:
        raise ValueError("Invalid sampler settings")
    if diffusion_scale < 0:
        raise ValueError("Diffusion scale cannot be negative")
    brownian = torch.Generator(device=x.device).manual_seed(seed)
    derivative_rng = torch.Generator(device=x.device).manual_seed(seed + 1)
    x = center(x.detach().float(), batch)
    dt = terminal_time / steps
    frames = []
    stochastic = mode == "sde" and diffusion_scale > 0
    for k in range(steps):
        t = x.new_tensor(k * dt)
        diffusion = diffusion_scale * (1 - t)
        with torch.enable_grad() if stochastic else torch.no_grad():
            state = x.detach().requires_grad_(stochastic)
            v, g = predict(state, t)
            v, g = v.float(), g.float()
            if stochastic:
                score = score_from_velocity(state, t, v, source_scale)
                div = projected_divergence(g, state, batch, probes, derivative_rng)
                drift = v + 0.5 * diffusion * (div + covariance_apply(g, score, batch))
            else:
                drift = v
            if save_trajectory and (k % max(1, steps // 5) == 0):
                frames.append(
                    {
                        "t": float(t),
                        "endpoint_mean": (state + (1 - t) * v).detach().cpu(),
                        "endpoint_variance": ((1 - t).square() * marginal_variance(g, batch))
                        .detach()
                        .cpu(),
                    }
                )
            delta = dt * drift
            if stochastic:
                eps = torch.randn(x.shape, device=x.device, generator=brownian)
                delta = delta + (diffusion * dt) ** 0.5 * noise_from_standard(g, eps, batch)
            x = center(state + delta, batch).detach()
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"Nonfinite PFM state at step {k}")
    # Both samplers use the same finite-time endpoint-mean completion.
    with torch.no_grad():
        t = x.new_tensor(terminal_time)
        v, g = predict(x, t)
        final = center(x + (1 - t) * v.float(), batch)
        frames.append(
            {
                "t": terminal_time,
                "endpoint_mean": final.cpu(),
                "endpoint_variance": ((1 - t).square() * marginal_variance(g.float(), batch)).cpu(),
            }
        )
    if not torch.isfinite(final).all():
        raise FloatingPointError("Nonfinite endpoint completion")
    return final, frames
