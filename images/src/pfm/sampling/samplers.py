"""One sampler for Euler ODE and corrected Itô Euler--Maruyama PFM."""
from contextlib import nullcontext
import torch
from pfm.probability import posterior, score_from_velocity, sde_coefficients
from pfm.sampling.randomness import indexed_step_random


def predict(model, x, t, labels, config, precision="fp32"):
    with torch.autocast("cuda", dtype=torch.bfloat16) if x.device.type == "cuda" and precision == "bf16" else nullcontext():
        outputs = model(x, t.reshape(-1), labels)
    return posterior(outputs, x, t, config)


def variance_divergence(gamma, x, probes, count):
    """Rademacher VJPs estimate diag(d gamma / dx), including the variance map."""
    mean = torch.zeros_like(x)
    if not gamma.requires_grad:
        return mean
    for index, z in enumerate(probes):
        vjp = torch.autograd.grad(gamma, x, z, retain_graph=index + 1 < count,
                                  create_graph=False, allow_unused=True)[0]
        value = torch.zeros_like(x) if vjp is None else z * vjp
        mean += (value - mean) / (index + 1)
    return mean.detach()


@torch.no_grad()
def sample(model, labels, config, sampling, device, *, seed=0, initial_noise=None,
           sample_indices=None, capture_time=None):
    model.eval()
    device = torch.device(device)
    if device.type == "cpu" and sampling.precision != "fp32":
        raise ValueError("CPU sampling uses precision=fp32")
    labels = labels.to(device)
    shape = (len(labels), *config.image_shape)
    generator = torch.Generator(device=device).manual_seed(seed)
    x = config.noise_scale * torch.randn(shape, device=device, generator=generator) if initial_noise is None else initial_noise.detach().to(device).float().clone()
    indices = torch.arange(len(labels)) if sample_indices is None else sample_indices
    h = sampling.terminal_time / sampling.steps
    capture_step = sampling.steps // 2 if capture_time is None else round(capture_time / h)
    if capture_step < 0 or capture_step >= sampling.steps or (capture_time is not None and abs(capture_step * h - capture_time) > 1e-10):
        raise ValueError("capture_time must lie on the sampling grid before terminal_time")
    captured = {}
    for step in range(sampling.steps):
        t_value = step * sampling.terminal_time / sampling.steps
        t = torch.full((len(labels), 1, 1, 1), t_value, device=device)
        lam = sampling.lambda_at(t_value) if sampling.solver == "sde" else 0.0
        if lam > 0:
            with torch.enable_grad():
                current = x.detach().requires_grad_(True)
                prediction = predict(model, current, t, labels, config, sampling.precision)
                probes = (indexed_step_random(indices, shape[1:], seed=seed, step=step,
                          stream="derivative", probe=p, device=device) for p in range(sampling.derivative_probes))
                divergence = variance_divergence(prediction.velocity_var, current, probes, sampling.derivative_probes)
            gamma, velocity = prediction.velocity_var.detach(), prediction.velocity.detach()
            score = score_from_velocity(x, t, velocity, config.noise_scale)
            drift, rate = sde_coefficients(velocity, gamma, score, divergence, lam)
            xi = indexed_step_random(indices, shape[1:], seed=seed, step=step, stream="brownian", device=device)
            next_x = x + h * drift + (h * rate).sqrt() * xi
        else:
            prediction = predict(model, x, t, labels, config, sampling.precision)
            next_x = x + h * prediction.velocity
        if step == capture_step:
            captured = {"velocity_variance": prediction.velocity_var.detach().cpu(),
                        "endpoint_variance": prediction.endpoint_var.detach().cpu(),
                        "capture_time": t_value}
        x = next_x.detach()
    if sampling.terminal_readout == "endpoint_mean":
        t = torch.full((len(labels), 1, 1, 1), sampling.terminal_time, device=device)
        x = predict(model, x, t, labels, config, sampling.precision).endpoint_mean.detach()
    if not torch.isfinite(x).all():
        raise FloatingPointError("sampling produced nonfinite values")
    return {"samples": x.cpu(), "labels": labels.cpu(), **captured}


def initial_noise(config, indices, seed):
    """Native fixed-block initialization, shared by samples and metric batches."""
    indices = indices.cpu()
    result = torch.empty((len(indices), *config.image_shape))
    block_size = 64
    for block in torch.unique(indices // block_size).tolist():
        rng = torch.Generator().manual_seed((seed + 1_000_003 * block) % (2**63 - 1))
        values = config.noise_scale * torch.randn((block_size, *config.image_shape), generator=rng)
        selected = indices // block_size == block
        result[selected] = values[indices[selected] % block_size]
    return result
