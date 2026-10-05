import torch
import pytest

from pfm.core.config import ExperimentConfig
from pfm.probability.logvar import inverse_smooth_logvar_bound
from pfm.probability.logvar import bound_normalized_logvar
from pfm.training.objectives import compute_loss


def _batch(batch_size=3, channels=2, image_size=2):
    torch.manual_seed(0)
    x0 = torch.randn(batch_size, channels, image_size, image_size)
    x1 = torch.randn(batch_size, channels, image_size, image_size)
    t = torch.linspace(0.1, 0.9, batch_size).view(batch_size, 1, 1, 1)
    xt = (1.0 - t) * x0 + t * x1
    return x0, x1, xt, t


def test_endpoint_fm_keeps_velocity_space_residual():
    cfg = ExperimentConfig(image_channels=2, image_size=2)
    x0, x1, xt, t = _batch(channels=cfg.image_channels, image_size=cfg.image_size)
    prediction = torch.randn_like(x1)

    endpoint_loss, _ = compute_loss(
        "endpoint_fm",
        {"x1_hat": prediction},
        x0,
        x1,
        xt,
        t,
        cfg,
    )
    residual = (x1 - xt) / (1.0 - t) - (prediction - xt) / (1.0 - t)
    expected_loss = 0.5 * residual.pow(2).mean()

    assert torch.allclose(endpoint_loss, expected_loss)


def test_all_variant_losses_are_finite():
    cfg = ExperimentConfig(image_channels=2, image_size=2)
    x0, x1, xt, t = _batch(channels=cfg.image_channels, image_size=cfg.image_size)
    outputs_by_variant = {
        "pfm_full": {"mu": torch.randn_like(x1), "logvar": torch.randn_like(x1)},
        "pfm_delta": {"delta": torch.randn_like(x1), "logvar": torch.randn_like(x1)},
        "endpoint_fm": {"x1_hat": torch.randn_like(x1)},
        "velocity_fm": {"velocity": torch.randn_like(x1)},
    }

    for variant, outputs in outputs_by_variant.items():
        loss, stats = compute_loss(variant, outputs, x0, x1, xt, t, cfg)
        assert torch.isfinite(loss)
        assert loss.ndim == 0
        assert "loss" in stats
        assert all(
            isinstance(value, torch.Tensor)
            and value.ndim == 0
            and value.device == loss.device
            and not value.requires_grad
            for value in stats.values()
        )


def test_pfm_delta_loss_is_finite_and_reports_diagonal_stats_only():
    cfg = ExperimentConfig(image_channels=2, image_size=2)
    x0, x1, xt, t = _batch(channels=cfg.image_channels, image_size=cfg.image_size)
    outputs = {
        "delta": torch.randn_like(x1),
        "logvar": torch.randn_like(x1),
    }

    loss, stats = compute_loss("pfm_delta", outputs, x0, x1, xt, t, cfg)

    assert torch.isfinite(loss)
    assert "covariance_" + "rank" not in stats
    assert "low_" + "rank_logdet_mean" not in stats
    assert torch.isfinite(stats["velocity_variance_mean"])
    assert bool(stats["velocity_variance_mean"] > 0.0)
    assert not any(key.startswith("covariance_") for key in stats)


def test_velocity_fm_loss_uses_direct_velocity_target_at_terminal_time():
    cfg = ExperimentConfig(image_channels=2, image_size=2)
    x0, x1, _, _ = _batch(channels=cfg.image_channels, image_size=cfg.image_size)
    t = torch.ones(x1.shape[0], 1, 1, 1)
    xt = x1

    loss, stats = compute_loss("velocity_fm", {"velocity": x1 - x0}, x0, x1, xt, t, cfg)

    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.zeros(()))
    assert torch.equal(stats["residual_mse"], torch.zeros(()))


def test_pfm_delta_zero_delta_matches_zero_velocity_fm_when_logvar_init_is_zero():
    cfg = ExperimentConfig(image_channels=2, image_size=2, logvar_init=0.0)
    x0, x1, xt, t = _batch(channels=cfg.image_channels, image_size=cfg.image_size)

    pfm_loss, pfm_stats = compute_loss(
        "pfm_delta",
        {
            "delta": torch.zeros_like(xt),
            "logvar": torch.full_like(xt, inverse_smooth_logvar_bound(cfg.logvar_init, cfg)),
        },
        x0,
        x1,
        xt,
        t,
        cfg,
    )
    velocity_loss, velocity_stats = compute_loss(
        "velocity_fm",
        {"velocity": torch.zeros_like(x1)},
        x0,
        x1,
        xt,
        t,
        cfg,
    )

    assert torch.allclose(pfm_loss, velocity_loss)
    assert torch.equal(
        pfm_stats["residual_mse"],
        velocity_stats["residual_mse"],
    )


def test_pfm_delta_loss_uses_direct_delta_target_at_terminal_time():
    cfg = ExperimentConfig(image_channels=2, image_size=2, logvar_init=0.0)
    x0, x1, _, _ = _batch(channels=cfg.image_channels, image_size=cfg.image_size)
    t = torch.ones(x1.shape[0], 1, 1, 1)
    xt = x1
    raw_logvar = torch.full_like(x1, inverse_smooth_logvar_bound(cfg.logvar_init, cfg))

    loss, stats = compute_loss("pfm_delta", {"delta": x1 - x0, "logvar": raw_logvar}, x0, x1, xt, t, cfg)

    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.zeros(()))
    assert torch.equal(stats["residual_mse"], torch.zeros(()))


def test_normalized_logvar_bound_is_smooth_and_within_config_range():
    cfg = ExperimentConfig(image_channels=2, image_size=2, logvar_min=-5.0, logvar_max=2.0, logvar_init=0.0)
    raw = torch.tensor([[-10.0, 0.0, 10.0]], requires_grad=True)

    bounded = bound_normalized_logvar(raw, cfg)
    bounded.sum().backward()

    assert torch.all(bounded > cfg.logvar_min)
    assert torch.all(bounded < cfg.logvar_max)
    assert torch.all(torch.isfinite(raw.grad))
    assert torch.all(raw.grad > 0)


def test_bfloat16_logvar_bound_uses_fp32_probability_math_and_retains_gradient():
    cfg = ExperimentConfig(
        image_channels=1,
        image_size=1,
        logvar_min=-5.0,
        logvar_max=2.0,
    )
    raw = torch.tensor([4.0, 5.0, 6.0, 7.0, 10.0], dtype=torch.bfloat16, requires_grad=True)

    bounded = bound_normalized_logvar(raw, cfg)
    bounded.sum().backward()

    assert bounded.dtype == torch.float32
    assert torch.all(bounded > cfg.logvar_min)
    assert torch.all(bounded < cfg.logvar_max)
    assert raw.grad is not None
    assert torch.all(torch.isfinite(raw.grad))
    assert torch.all(raw.grad > 0)


@pytest.mark.parametrize("variant", ["pfm_full", "pfm_delta"])
def test_pfm_loss_uses_fp32_logvar_probability_reference_under_bfloat16_outputs(variant):
    cfg = ExperimentConfig(
        image_channels=1,
        image_size=2,
        logvar_min=-5.0,
        logvar_max=2.0,
    )
    x0, x1, xt, t = _batch(
        batch_size=2,
        channels=cfg.image_channels,
        image_size=cfg.image_size,
    )
    raw_logvar = torch.full(
        x1.shape,
        7.0,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    if variant == "pfm_delta":
        prediction = torch.zeros(x1.shape, dtype=torch.bfloat16)
        outputs = {"delta": prediction, "logvar": raw_logvar}
        residual = (x1 - x0) - prediction.float()
    else:
        prediction = torch.zeros(x1.shape, dtype=torch.bfloat16)
        outputs = {"mu": prediction, "logvar": raw_logvar}
        target = (x1 - xt) / (1.0 - t)
        predicted_velocity = (prediction.float() - xt) / (1.0 - t)
        residual = target - predicted_velocity
    reference_logvar = cfg.logvar_min + (cfg.logvar_max - cfg.logvar_min) * torch.sigmoid(
        raw_logvar.float()
    )
    reference_loss = 0.5 * (
        torch.exp(-reference_logvar) * residual.float().square() + reference_logvar
    ).flatten(1).mean(dim=1).mean()

    loss, stats = compute_loss(variant, outputs, x0, x1, xt, t, cfg)
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.allclose(loss, reference_loss, rtol=1e-6, atol=1e-6)
    assert torch.allclose(
        stats["logvar_mean"],
        reference_logvar.detach().mean(),
        rtol=1e-6,
        atol=1e-6,
    )
    assert raw_logvar.grad is not None
    assert torch.all(torch.isfinite(raw_logvar.grad))
    assert torch.all(raw_logvar.grad != 0)


@pytest.mark.parametrize("variant", ["velocity_fm", "endpoint_fm"])
def test_baseline_residual_mse_is_reduced_once(monkeypatch, variant):
    cfg = ExperimentConfig(image_channels=2, image_size=2)
    x0, x1, xt, t = _batch(channels=cfg.image_channels, image_size=cfg.image_size)
    outputs = {
        "velocity_fm": {"velocity": torch.randn_like(x1)},
        "endpoint_fm": {"x1_hat": torch.randn_like(x1)},
    }[variant]
    original_mean = torch.Tensor.mean
    mean_calls = 0

    def counting_mean(tensor, *args, **kwargs):
        nonlocal mean_calls
        mean_calls += 1
        return original_mean(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "mean", counting_mean)

    loss, stats = compute_loss(variant, outputs, x0, x1, xt, t, cfg)

    assert mean_calls == 1
    assert torch.allclose(stats["residual_mse"], 2.0 * loss.detach())


def test_logvar_boundary_init_uses_finite_interior_proxy():
    cfg_low = ExperimentConfig(image_channels=2, image_size=2, logvar_min=0.0, logvar_max=5.0, logvar_init=0.0)
    raw_low = inverse_smooth_logvar_bound(cfg_low.logvar_init, cfg_low)
    bounded_low = bound_normalized_logvar(torch.tensor(raw_low), cfg_low)

    assert torch.isfinite(torch.tensor(raw_low))
    assert cfg_low.logvar_min < float(bounded_low) < cfg_low.logvar_max
    assert abs(float(bounded_low) - cfg_low.logvar_min) <= 6e-4

    cfg_high = ExperimentConfig(image_channels=2, image_size=2, logvar_min=0.0, logvar_max=5.0, logvar_init=5.0)
    raw_high = inverse_smooth_logvar_bound(cfg_high.logvar_init, cfg_high)
    bounded_high = bound_normalized_logvar(torch.tensor(raw_high), cfg_high)

    assert torch.isfinite(torch.tensor(raw_high))
    assert cfg_high.logvar_min < float(bounded_high) < cfg_high.logvar_max
    assert abs(float(bounded_high) - cfg_high.logvar_max) <= 6e-4
