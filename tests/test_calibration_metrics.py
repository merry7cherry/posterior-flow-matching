import math

import pytest
import torch

from pfm.diagnostics import calibration_metrics as calibration_metrics_module
from pfm.diagnostics import posterior_residuals as posterior_residuals_module
from pfm.diagnostics.calibration_metrics import (
    evaluate_pfm_calibration,
    summarize_time_calibration,
    summarize_variance_calibration,
)
from pfm.diagnostics.posterior_diagnostics import (
    evaluate_controlled_interpolant_posterior,
)
from pfm.core.config import ExperimentConfig
from pfm.diagnostics.posterior_residuals import pfm_residual_tensors


def test_variance_calibration_summary_matches_equal_count_bins():
    predicted_variance = torch.tensor([1.0, 1.0, 4.0, 4.0])
    squared_error = torch.tensor([1.0, 1.0, 4.0, 4.0])

    summary = summarize_variance_calibration(predicted_variance, squared_error, num_bins=2)

    assert summary["ece_abs"] == pytest.approx(0.0)
    assert summary["ece_rel"] == pytest.approx(0.0)
    assert summary["bins"][0]["mean_pred_var"] == pytest.approx(1.0)
    assert summary["bins"][0]["mean_squared_error"] == pytest.approx(1.0)
    assert summary["bins"][1]["mean_pred_var"] == pytest.approx(4.0)
    assert summary["bins"][1]["mean_squared_error"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("predicted_variance", "squared_error"),
    [
        ([1.0, float("nan"), 3.0], [1.0, 2.0, 3.0]),
        ([1.0, float("inf"), 3.0], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [1.0, float("nan"), 3.0]),
        ([1.0, 2.0, 3.0], [1.0, float("inf"), 3.0]),
        ([1.0, 0.0, 3.0], [1.0, 2.0, 3.0]),
        ([1.0, -1.0, 3.0], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [1.0, -1.0, 3.0]),
    ],
)
def test_variance_calibration_rejects_mixed_invalid_population(
    predicted_variance,
    squared_error,
):
    with pytest.raises(ValueError):
        summarize_variance_calibration(
            torch.tensor(predicted_variance),
            torch.tensor(squared_error),
            num_bins=2,
        )


def test_variance_calibration_rejects_incompatible_population_shapes():
    with pytest.raises(ValueError):
        summarize_variance_calibration(
            torch.ones(2, 2),
            torch.ones(4),
            num_bins=2,
        )


@pytest.mark.parametrize(
    "invalid_time",
    [float("nan"), float("inf"), float("-inf"), -0.1, 1.1],
)
def test_time_calibration_rejects_mixed_invalid_time_population(invalid_time):
    with pytest.raises(ValueError):
        summarize_time_calibration(
            torch.tensor([0.1, invalid_time, 0.7]),
            torch.ones(3),
            torch.ones(3),
            num_bins=2,
        )


def test_valid_calibration_counts_reconcile_to_complete_population():
    predicted_variance = torch.tensor([1.0, 1.0, 4.0, 9.0])
    squared_error = torch.tensor([1.0, 2.0, 4.0, 8.0])
    times = torch.tensor([0.05, 0.15, 0.55, 0.65])

    variance_summary = summarize_variance_calibration(
        predicted_variance,
        squared_error,
        num_bins=2,
    )
    time_summary = summarize_time_calibration(
        times,
        predicted_variance,
        squared_error,
        num_bins=2,
    )

    assert variance_summary["count"] == predicted_variance.numel()
    assert (
        sum(item["count"] for item in variance_summary["bins"])
        == variance_summary["count"]
    )
    assert (
        sum(item["count"] for item in variance_summary["histogram_bins"])
        == variance_summary["count"]
    )
    assert time_summary["count"] == predicted_variance.numel()
    assert (
        sum(item["count"] for item in time_summary["bins"])
        == time_summary["count"]
    )


def test_variance_calibration_summary_uses_fixed_width_bins_for_histogram():
    predicted_variance = torch.tensor([1.0, 1.0, 1.0, 9.0])
    squared_error = torch.ones_like(predicted_variance)

    summary = summarize_variance_calibration(predicted_variance, squared_error, num_bins=2)

    assert [item["count"] for item in summary["bins"]] == [3, 1]
    assert summary["histogram_binning"] == "fixed_shared_edges"
    assert [item["count"] for item in summary["histogram_bins"]] == [3, 1]
    assert summary["histogram_bins"][0]["min_pred_var"] == pytest.approx(1.0)
    assert summary["histogram_bins"][0]["max_pred_var"] == pytest.approx(5.0)


def test_variance_histogram_preserves_explicit_floating_edge_tolerance():
    edges = torch.tensor([1.0, 1.5, 2.0], dtype=torch.float64)
    within_tolerance = torch.tensor(
        [1.0 - 1e-6, 2.0 + 1e-6],
        dtype=torch.float64,
    )

    summary = summarize_variance_calibration(
        within_tolerance,
        torch.ones(2, dtype=torch.float64),
        num_bins=2,
        histogram_edges=edges,
    )

    assert summary["count"] == 2
    assert sum(item["count"] for item in summary["histogram_bins"]) == 2
    assert summary["mean_pred_var"] == pytest.approx(within_tolerance.mean().item())

    with pytest.raises(ValueError, match="outside the shared histogram edges"):
        summarize_variance_calibration(
            torch.tensor([1.0 - 3e-6, 1.5], dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            num_bins=2,
            histogram_edges=edges,
        )


def test_variance_calibration_penalizes_overconfident_variance():
    matched = summarize_variance_calibration(torch.ones(4), torch.ones(4), num_bins=2)
    overconfident = summarize_variance_calibration(torch.full((4,), 0.1), torch.ones(4), num_bins=2)

    assert overconfident["ece_abs"] > matched["ece_abs"]
    assert overconfident["mean_pred_var"] < overconfident["mean_squared_error"]


def test_time_calibration_summary_bins_by_time():
    times = torch.tensor([0.05, 0.15, 0.55, 0.65])
    predicted_variance = torch.tensor([1.0, 3.0, 5.0, 7.0])
    squared_error = torch.tensor([2.0, 4.0, 10.0, 14.0])

    summary = summarize_time_calibration(times, predicted_variance, squared_error, num_bins=2)

    assert summary["time_binning"] == "equal_width_bounded_interval"
    assert summary["time_bounds"] == [0.0, 1.0]
    assert summary["num_bins"] == 2
    assert summary["bins"][0]["t_min"] == pytest.approx(0.05)
    assert summary["bins"][0]["t_max"] == pytest.approx(0.15)
    assert summary["bins"][0]["t_mean"] == pytest.approx(0.10)
    assert summary["bins"][0]["count"] == 2
    assert summary["bins"][0]["mean_pred_var"] == pytest.approx(2.0)
    assert summary["bins"][0]["mean_squared_error"] == pytest.approx(3.0)
    assert summary["bins"][0]["error_to_var_ratio"] == pytest.approx(1.5)
    assert summary["bins"][1]["t_mean"] == pytest.approx(0.60)
    assert summary["bins"][1]["mean_pred_var"] == pytest.approx(6.0)
    assert summary["bins"][1]["mean_squared_error"] == pytest.approx(12.0)


def test_time_calibration_preserves_explicit_floating_boundary_tolerance():
    tolerance_offset = 0.5e-12
    times = torch.tensor(
        [0.1 - tolerance_offset, 0.9 + tolerance_offset],
        dtype=torch.float64,
    )

    summary = summarize_time_calibration(
        times,
        torch.ones(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        num_bins=2,
        t_min=0.1,
        t_max=0.9,
    )

    assert summary["count"] == 2
    assert sum(item["count"] for item in summary["bins"]) == 2
    assert summary["bins"][0]["t_min"] == pytest.approx(times[0].item())
    assert summary["bins"][-1]["t_max"] == pytest.approx(times[1].item())


def test_pfm_delta_residual_tensors_use_velocity_space_target_and_induced_endpoint_variance():
    cfg = ExperimentConfig(image_channels=1, image_size=2, logvar_min=-1.0, logvar_max=1.0)
    x0 = torch.zeros(1, 1, 2, 2)
    x1 = torch.ones(1, 1, 2, 2)
    t = torch.full((1, 1, 1, 1), 0.25)
    xt = (1.0 - t) * x0 + t * x1
    outputs = {"delta": torch.full_like(x1, 0.5), "logvar": torch.zeros_like(x1)}

    tensors = pfm_residual_tensors("pfm_delta", outputs, x0, x1, xt, t, cfg)

    assert torch.allclose(tensors["velocity_squared_error"], torch.full_like(x1, 0.25))
    assert torch.allclose(tensors["velocity_variance"], torch.ones_like(x1))
    assert torch.allclose(tensors["endpoint_variance"], torch.full_like(x1, 0.75**2))
    assert torch.allclose(tensors["endpoint_squared_error"], torch.full_like(x1, (1.0 - (0.25 + 0.75 * 0.5)) ** 2))
    assert "endpoint_nll" not in tensors
    assert "velocity_nll" not in tensors


def test_diagonal_residual_tensors_name_constant_free_and_complete_nll_units():
    cfg = ExperimentConfig(image_channels=1, image_size=2, logvar_min=-1.0, logvar_max=1.0)
    x0 = torch.zeros(1, 1, 2, 2)
    x1 = torch.ones(1, 1, 2, 2)
    t = torch.full((1, 1, 1, 1), 0.25)
    xt = (1.0 - t) * x0 + t * x1
    outputs = {"delta": torch.zeros_like(x1), "logvar": torch.zeros_like(x1)}

    tensors = pfm_residual_tensors("pfm_delta", outputs, x0, x1, xt, t, cfg)

    expected_velocity = torch.full((1,), 0.5)
    expected_endpoint = expected_velocity + math.log(0.75)
    gaussian_constant = 0.5 * math.log(2.0 * math.pi)
    num_components = x1[0].numel()

    assert torch.allclose(tensors["velocity_constant_free_nll_per_dim"], expected_velocity)
    assert torch.allclose(tensors["endpoint_constant_free_nll_per_dim"], expected_endpoint)
    assert torch.allclose(
        tensors["velocity_complete_nll_per_dim"],
        expected_velocity + gaussian_constant,
    )
    assert torch.allclose(
        tensors["endpoint_complete_nll_per_dim"],
        expected_endpoint + gaussian_constant,
    )
    assert torch.allclose(
        tensors["endpoint_complete_nll_total"],
        tensors["endpoint_complete_nll_per_dim"] * num_components,
    )


def test_posterior_diagnostics_direct_calls_reject_baseline_variants():
    config = ExperimentConfig(dataset="mnist")
    loader = []
    model = torch.nn.Identity()
    with pytest.raises(ValueError, match="requires learned variance"):
        evaluate_pfm_calibration(
            model, "velocity_fm", config, loader, torch.device("cpu"), num_samples=1
        )
    with pytest.raises(ValueError, match="requires learned variance"):
        evaluate_controlled_interpolant_posterior(
            model, "endpoint_fm", config, loader, torch.device("cpu"), num_samples=1
        )


def test_diagonal_calibration_exports_split_sections_without_factor_fields():
    cfg = ExperimentConfig(image_channels=1, image_size=2)

    class DiagonalDeltaModel(torch.nn.Module):
        def forward(self, x, t, labels):
            del t, labels
            return {
                "mu": x,
                "delta": torch.zeros_like(x),
                "logvar": torch.zeros_like(x),
            }

    data_loader = [(torch.ones(2, 1, 2, 2), torch.zeros(2, dtype=torch.long))]

    result = evaluate_pfm_calibration(
        DiagonalDeltaModel(),
        "pfm_delta",
        cfg,
        data_loader,
        torch.device("cpu"),
        num_samples=2,
        num_bins=2,
    )

    assert result["available"] is True
    assert "covariance_" + "type" not in result
    assert "covariance_" + "rank" not in result
    assert "variance_calibration" in result
    assert "time_calibration" in result
    assert "variance_nll" in result
    assert "covariance_diagnostics" not in result
    assert "velocity" not in result
    assert "endpoint" not in result
    assert "covariance" not in result
    assert result["num_samples"] == 2
    for space in ("velocity", "endpoint"):
        variance_summary = result["variance_calibration"][space]
        time_summary = result["time_calibration"][space]
        assert variance_summary["count"] == 2
        assert sum(item["count"] for item in variance_summary["bins"]) == 2
        assert sum(item["count"] for item in variance_summary["histogram_bins"]) == 2
        assert time_summary["count"] == 2
        assert sum(item["count"] for item in time_summary["bins"]) == 2


def test_random_time_calibration_streams_raw_logvar_saturation_by_time(monkeypatch):
    cfg = ExperimentConfig(image_channels=1, image_size=2, eps_t=0.1)

    class SaturatedDeltaModel(torch.nn.Module):
        def forward(self, x, t, labels):
            del t, labels
            raw_logvar = torch.empty_like(x)
            raw_logvar[..., 0] = -10.0
            raw_logvar[..., 1] = 10.0
            return {
                "delta": torch.zeros_like(x),
                "logvar": raw_logvar,
            }

    sampled_times = iter(
        (
            torch.tensor([0.2]),
            torch.tensor([0.8]),
        )
    )
    monkeypatch.setattr(
        calibration_metrics_module,
        "sample_train_timesteps",
        lambda config, batch_size, device, generator=None: next(sampled_times).to(device),
    )
    data_loader = [
        (torch.zeros(1, 1, 2, 2), torch.zeros(1, dtype=torch.long)),
        (torch.zeros(1, 1, 2, 2), torch.zeros(1, dtype=torch.long)),
    ]

    result = evaluate_pfm_calibration(
        SaturatedDeltaModel(),
        "pfm_delta",
        cfg,
        data_loader,
        torch.device("cpu"),
        num_samples=2,
        num_bins=2,
    )

    saturation = result["raw_logvar_saturation"]
    assert saturation["component_count"] == 8
    assert saturation["lower_count"] == 4
    assert saturation["upper_count"] == 4
    assert saturation["lower_rate"] == pytest.approx(0.5)
    assert saturation["upper_rate"] == pytest.approx(0.5)
    assert saturation["time_binning"] == "equal_width_bounded_interval"
    assert [item["sample_count"] for item in saturation["time_bins"]] == [1, 1]
    assert sum(item["component_count"] for item in saturation["time_bins"]) == 8


def test_controlled_posterior_counts_reconcile_at_every_time():
    cfg = ExperimentConfig(image_channels=1, image_size=2)

    class DiagonalDeltaModel(torch.nn.Module):
        def forward(self, x, t, labels):
            del t, labels
            return {
                "delta": torch.zeros_like(x),
                "logvar": torch.zeros_like(x),
            }

    data_loader = [(torch.ones(2, 1, 2, 2), torch.zeros(2, dtype=torch.long))]

    result = evaluate_controlled_interpolant_posterior(
        DiagonalDeltaModel(),
        "pfm_delta",
        cfg,
        data_loader,
        torch.device("cpu"),
        num_samples=2,
        t_values=(0.25, 0.75),
        num_bins=2,
    )

    assert result["num_samples"] == 2
    for payload in result["by_t"].values():
        assert "num_samples" not in payload
        for space in ("velocity", "endpoint"):
            summary = payload[space]
            assert summary["count"] == 2
            assert sum(item["count"] for item in summary["bins"]) == 2
            assert sum(item["count"] for item in summary["histogram_bins"]) == 2


def test_posterior_calibration_uses_training_noise_scale(monkeypatch):
    cfg = ExperimentConfig(image_channels=1, image_size=2, noise_scale=4.0)
    seen_xt = []

    class RecordingDeltaModel(torch.nn.Module):
        def forward(self, x, t, labels):
            del t, labels
            seen_xt.append(x.detach().clone())
            return {
                "delta": torch.zeros_like(x),
                "logvar": torch.zeros_like(x),
            }

    monkeypatch.setattr(
        posterior_residuals_module,
        "_randn_like",
        lambda x, generator: torch.ones_like(x),
    )
    monkeypatch.setattr(
        calibration_metrics_module,
        "sample_train_timesteps",
        lambda config, batch_size, device, generator=None: torch.full(
            (batch_size, 1, 1, 1),
            0.5,
            device=device,
        ),
    )
    data_loader = [(torch.zeros(2, 1, 2, 2), torch.zeros(2, dtype=torch.long))]

    evaluate_pfm_calibration(
        RecordingDeltaModel(),
        "pfm_delta",
        cfg,
        data_loader,
        torch.device("cpu"),
        num_samples=2,
        num_bins=2,
    )

    assert torch.allclose(seen_xt[0], torch.full_like(seen_xt[0], 2.0))


def test_posterior_residuals_use_one_stateless_forward_per_batch(monkeypatch):
    cfg = ExperimentConfig(image_channels=1, image_size=2)
    calls = []

    class RecordingFullModel(torch.nn.Module):
        def forward(self, x, t, labels):
            calls.append((x.shape, t.shape, labels.shape))
            return {"mu": x, "logvar": torch.zeros_like(x)}

    monkeypatch.setattr(
        posterior_residuals_module,
        "_randn_like",
        lambda x, generator: torch.ones_like(x),
    )
    batch = (torch.zeros(2, 1, 2, 2), torch.zeros(2, dtype=torch.long))
    result = posterior_residuals_module.collect_posterior_residual_statistics(
        model=RecordingFullModel(),
        variant="pfm_full",
        config=cfg,
        data_loader=[batch, batch],
        device=torch.device("cpu"),
        num_samples=3,
        num_bins=2,
        t_sampler=lambda config, batch_size, device, variant, dtype: torch.full(
            (batch_size, 1, 1, 1), 0.5, device=device, dtype=dtype,
        ),
        include_time_calibration=False,
    )
    assert len(calls) == 2
    assert calls[0][1] == (2,)
    assert calls[1][1] == (1,)
    assert result["actual_nfe_per_batch"] == 1
    assert result["num_samples"] == 3
    assert "self_conditioning" not in result
    assert "self_conditioning_forward_policy" not in result
    assert result["standardized_residual"]["component_count"] == 12
    for summary in result["variance_calibration"].values():
        assert summary["count"] == 3
        assert sum(item["count"] for item in summary["bins"]) == 3
        assert sum(item["count"] for item in summary["histogram_bins"]) == 3


def test_random_time_posterior_rejects_component_time_outside_authored_interval():
    cfg = ExperimentConfig(image_channels=1, image_size=2, eps_t=0.1)
    data_loader = [(torch.zeros(1, 1, 2, 2), torch.zeros(1, dtype=torch.long))]

    class ModelMustNotRun(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("sampled time must be rejected before model execution")

    with pytest.raises(ValueError, match="outside the authored bounded interval"):
        posterior_residuals_module.collect_posterior_residual_statistics(
            model=ModelMustNotRun(),
            variant="pfm_full",
            config=cfg,
            data_loader=data_loader,
            device=torch.device("cpu"),
            num_samples=1,
            num_bins=2,
            t_sampler=lambda config, batch_size, device, variant, dtype: torch.tensor(
                [[[[0.0, 0.2], [0.2, 0.2]]]],
                device=device,
                dtype=dtype,
            ),
            include_time_calibration=True,
        )


def test_non_pfm_variants_do_not_report_calibration_tensors():
    cfg = ExperimentConfig()

    with pytest.raises(ValueError, match="requires learned variance"):
        pfm_residual_tensors(
            "velocity_fm",
            {"velocity": torch.zeros(1, 1, 2, 2)},
            torch.zeros(1, 1, 2, 2),
            torch.ones(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 1, 1),
            cfg,
        )
