import math

import pytest
import torch

from pfm.core.config import ExperimentConfig
from pfm.diagnostics.posterior_residuals import pfm_residual_tensors, summarize_variance_calibration
from pfm.diagnostics.standardized_residuals import (
    GAUSSIAN_COVERAGE_THRESHOLDS,
    StandardizedResidualAccumulator,
)


def _summary(z, times, *, chunk_size, include_time=True):
    accumulator = StandardizedResidualAccumulator(include_time=include_time)
    for start in range(0, len(z), chunk_size):
        accumulator.update(z[start : start + chunk_size], times[start : start + chunk_size])
    return accumulator.summary(num_bins=2, t_min=0.0, t_max=1.0)


def test_heteroscedastic_ratio_and_standardized_second_moment_answer_different_questions():
    variance = torch.tensor([[1.0, 9.0]])
    residual = torch.tensor([[2.0, 0.0]])
    z = residual / variance.sqrt()
    calibration = summarize_variance_calibration(variance, residual.square())
    standardized = _summary(z, torch.tensor([[0.5]]), chunk_size=1, include_time=False)

    assert calibration["error_to_var_ratio"] == pytest.approx(0.4)
    assert standardized["mean"] == pytest.approx(1.0)
    assert standardized["second_moment"] == pytest.approx(2.0)
    assert standardized["gaussian_coverage_90"] == 0.5
    assert standardized["gaussian_coverage_95"] == 0.5


def test_component_moments_and_coverage_reconcile_across_batches_and_time_bins():
    z = torch.tensor([
        [-2.0, -1.8, -1.0, 0.0],
        [0.0, 1.0, 1.8, 2.0],
        [-3.0, -2.0, 2.0, 3.0],
    ])
    times = torch.tensor([[0.1], [0.25], [0.75]])
    whole = _summary(z, times, chunk_size=3)
    chunked = _summary(z, times, chunk_size=1)

    assert whole == chunked
    assert whole["component_count"] == 12
    # The streaming accumulator uses FP64; a parallel FP32 reduction can
    # retain platform-dependent cancellation error even for this zero mean.
    assert whole["mean"] == pytest.approx(float(z.double().mean()))
    assert whole["second_moment"] == pytest.approx(float(z.double().square().mean()))
    assert whole["coverage_count_90"] == 4
    assert whole["coverage_count_95"] == 6
    assert whole["shared_spaces"] == ["velocity", "endpoint"]
    assert [item["sample_count"] for item in whole["time_bins"]] == [2, 1]
    for key in ("component_count", "coverage_count_90", "coverage_count_95"):
        assert sum(item[key] for item in whole["time_bins"]) == whole[key]
    for key in ("mean", "second_moment", "gaussian_coverage_90", "gaussian_coverage_95"):
        weighted = sum(item[key] * item["component_count"] for item in whole["time_bins"])
        assert weighted / whole["component_count"] == pytest.approx(whole[key])


def test_coverage_uses_two_sided_gaussian_component_thresholds():
    for level, threshold in GAUSSIAN_COVERAGE_THRESHOLDS.items():
        probability = math.erf(threshold / math.sqrt(2.0))
        assert probability == pytest.approx(int(level) / 100.0, abs=1e-12)
    q90 = torch.tensor(GAUSSIAN_COVERAGE_THRESHOLDS["90"])
    q95 = torch.tensor(GAUSSIAN_COVERAGE_THRESHOLDS["95"])
    beyond95 = torch.nextafter(q95, torch.tensor(math.inf))
    z = torch.stack((-q90, q90, -q95, q95, -beyond95, beyond95)).view(1, -1)
    summary = _summary(z, torch.tensor([[0.5]]), chunk_size=1, include_time=False)
    assert summary["coverage_count_90"] == 2
    assert summary["coverage_count_95"] == 4


@pytest.mark.parametrize("variant", ["pfm_delta", "pfm_full"])
def test_one_standardized_residual_matches_endpoint_and_velocity_spaces(variant):
    config = ExperimentConfig(image_channels=1, image_size=2, logvar_min=-3.0, logvar_max=1.0)
    x0 = torch.tensor([[[[-1.0, 0.5], [0.0, 1.0]]]])
    x1 = torch.tensor([[[[1.0, -0.5], [2.0, 3.0]]]])
    t = torch.tensor([0.75]).view(1, 1, 1, 1)
    xt = (1.0 - t) * x0 + t * x1
    velocity_mean = torch.tensor([[[[0.5, -0.25], [1.0, 1.5]]]])
    endpoint_mean = xt + (1.0 - t) * velocity_mean
    raw = torch.tensor([[[[-2.0, -0.5], [0.5, 2.0]]]])
    outputs = {"delta": velocity_mean, "mu": endpoint_mean, "logvar": raw}
    tensors = pfm_residual_tensors(variant, outputs, x0, x1, xt, t, config)

    expected_velocity = (x1 - x0 - velocity_mean) / tensors["velocity_variance"].sqrt()
    expected_endpoint = (x1 - endpoint_mean) / tensors["endpoint_variance"].sqrt()
    torch.testing.assert_close(tensors["standardized_residual"], expected_velocity)
    torch.testing.assert_close(tensors["standardized_residual"], expected_endpoint)


@pytest.mark.parametrize("z", [torch.tensor([[float("nan")]]), torch.tensor([[float("inf")]])])
def test_nonfinite_standardized_residual_is_rejected(z):
    with pytest.raises(ValueError, match="only finite values"):
        _summary(z, torch.tensor([[0.5]]), chunk_size=1)


def test_time_bins_reject_component_times_that_are_not_a_single_image_time():
    with pytest.raises(ValueError, match="constant within each image"):
        _summary(torch.ones(1, 2), torch.tensor([[0.1, 0.9]]), chunk_size=1)
