from contextlib import contextmanager

import pytest
import torch

from pfm.core.config import ExperimentConfig
from pfm.diagnostics import posterior_residuals as posterior_residuals_module
from pfm.diagnostics.posterior_residuals import pfm_residual_tensors


_CANONICAL_RESIDUAL_KEYS = {
    "standardized_residual",
    "velocity_variance",
    "velocity_squared_error",
    "velocity_constant_free_nll_per_dim",
    "velocity_complete_nll_per_dim",
    "velocity_complete_nll_total",
    "endpoint_variance",
    "endpoint_squared_error",
    "endpoint_constant_free_nll_per_dim",
    "endpoint_complete_nll_per_dim",
    "endpoint_complete_nll_total",
}


def test_full_and_delta_share_one_canonical_posterior_math_result():
    config = ExperimentConfig(
        image_channels=1,
        image_size=2,
        logvar_min=-1.0,
        logvar_max=1.0,
    )
    x0 = torch.tensor([[[[-1.0, 0.5], [0.0, 1.0]]]])
    x1 = torch.tensor([[[[1.0, -0.5], [2.0, 3.0]]]])
    t = torch.full((1, 1, 1, 1), 0.25)
    xt = (1.0 - t) * x0 + t * x1
    velocity_mean = torch.tensor([[[[0.5, -0.25], [1.0, 1.5]]]])
    endpoint_mean = xt + (1.0 - t) * velocity_mean
    raw_logvar = torch.tensor([[[[-0.5, 0.0], [0.5, 1.0]]]])

    delta = pfm_residual_tensors(
        "pfm_delta",
        {"delta": velocity_mean, "logvar": raw_logvar},
        x0,
        x1,
        xt,
        t,
        config,
    )
    full = pfm_residual_tensors(
        "pfm_full",
        {"mu": endpoint_mean, "logvar": raw_logvar},
        x0,
        x1,
        xt,
        t,
        config,
    )

    assert set(delta) == _CANONICAL_RESIDUAL_KEYS
    assert set(full) == _CANONICAL_RESIDUAL_KEYS
    for key in sorted(_CANONICAL_RESIDUAL_KEYS):
        assert torch.allclose(delta[key], full[key])


def test_collection_autocasts_only_model_forward_and_keeps_statistics_fp32(monkeypatch):
    config = ExperimentConfig(
        image_channels=1,
        image_size=2,
        mixed_precision="bf16",
    )
    state = {"autocast_active": False, "math_calls": 0}

    @contextmanager
    def recording_autocast(received_config, received_device):
        assert received_config is config
        assert received_device == torch.device("cpu")
        assert not state["autocast_active"]
        state["autocast_active"] = True
        try:
            yield
        finally:
            state["autocast_active"] = False

    class RecordingDeltaModel(torch.nn.Module):
        def forward(self, x, t, labels):
            del t, labels
            assert state["autocast_active"]
            return {
                "delta": torch.zeros_like(x, dtype=torch.bfloat16),
                "logvar": torch.zeros_like(x, dtype=torch.bfloat16),
            }

    original_residual_tensors = posterior_residuals_module.pfm_residual_tensors

    def recording_residual_tensors(*args, **kwargs):
        assert not state["autocast_active"]
        tensors = original_residual_tensors(*args, **kwargs)
        assert all(tensor.dtype == torch.float32 for tensor in tensors.values())
        state["math_calls"] += 1
        return tensors

    monkeypatch.setattr(posterior_residuals_module, "autocast_context", recording_autocast)
    monkeypatch.setattr(
        posterior_residuals_module,
        "pfm_residual_tensors",
        recording_residual_tensors,
    )
    monkeypatch.setattr(
        posterior_residuals_module,
        "_randn_like",
        lambda x, generator: torch.zeros_like(x),
    )

    result = posterior_residuals_module.collect_posterior_residual_statistics(
        model=RecordingDeltaModel(),
        variant="pfm_delta",
        config=config,
        data_loader=[(torch.ones(1, 1, 2, 2), torch.zeros(1, dtype=torch.long))],
        device=torch.device("cpu"),
        num_samples=1,
        num_bins=1,
        t_sampler=lambda config, batch_size, device, variant, dtype: torch.full(
            (batch_size, 1, 1, 1),
            0.5,
            device=device,
            dtype=dtype,
        ),
        include_time_calibration=False,
    )

    assert state == {"autocast_active": False, "math_calls": 1}
    assert result["num_samples"] == 1


def test_collection_streams_component_totals_but_calibrates_on_image_scalars(monkeypatch):
    config = ExperimentConfig(
        image_channels=1,
        image_size=2,
        logvar_min=-1.0,
        logvar_max=1.0,
    )
    images = torch.tensor(
        [
            [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]],
            [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]],
        ]
    )
    labels = torch.zeros(images.shape[0], dtype=torch.long)

    class ZeroDeltaModel(torch.nn.Module):
        def forward(self, x, t, labels):
            del t, labels
            return {
                "delta": torch.zeros_like(x),
                "logvar": torch.zeros_like(x),
            }

    monkeypatch.setattr(posterior_residuals_module.torch, "randn_like", torch.zeros_like)
    original_summary = posterior_residuals_module.summarize_variance_calibration
    calibration_input_sizes = []

    def recording_summary(
        predicted_variance,
        squared_error,
        num_bins=15,
        *,
        histogram_edges=None,
    ):
        calibration_input_sizes.append(
            (int(predicted_variance.numel()), int(squared_error.numel()))
        )
        return original_summary(
            predicted_variance,
            squared_error,
            num_bins=num_bins,
            histogram_edges=histogram_edges,
        )

    monkeypatch.setattr(
        posterior_residuals_module,
        "summarize_variance_calibration",
        recording_summary,
    )

    result = posterior_residuals_module.collect_posterior_residual_statistics(
        model=ZeroDeltaModel(),
        variant="pfm_delta",
        config=config,
        data_loader=[(images, labels)],
        device=torch.device("cpu"),
        num_samples=2,
        num_bins=2,
        t_sampler=lambda config, batch_size, device, variant, dtype: torch.full(
            (batch_size, 1, 1, 1),
            0.5,
            device=device,
            dtype=dtype,
        ),
        include_time_calibration=True,
    )

    image_count = int(images.shape[0])
    component_count = int(images.numel())
    expected_velocity_error = float(images.square().mean())

    assert calibration_input_sizes == [(image_count, image_count), (image_count, image_count)]
    assert result["variance_calibration"]["velocity"]["count"] == image_count
    assert result["variance_calibration"]["velocity"]["component_count"] == component_count
    assert result["variance_calibration"]["velocity"]["mean_pred_var"] == 1.0
    assert result["variance_calibration"]["velocity"]["mean_squared_error"] == pytest.approx(
        expected_velocity_error
    )
    assert result["time_calibration"]["velocity"]["count"] == image_count
    assert "covariance_diagnostics" not in result
