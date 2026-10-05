import pytest
import torch
from copy import deepcopy
from pathlib import Path

import yaml

from pfm.cli.config_inputs import resolved_config_bytes
from pfm.core.config import ExperimentConfig
from pfm.core.evaluation_config import metric_evaluation_config_from_mapping
from pfm.diagnostics import posterior_diagnostics as posterior_diagnostics_module
from pfm.diagnostics import posterior_residuals as posterior_residuals_module
from pfm.diagnostics.standardized_residuals import StandardizedResidualAccumulator
from pfm.diagnostics.posterior_diagnostics import (
    evaluate_controlled_interpolant_posterior,
)


def test_cli_nearby_times_retain_independent_results_under_reordering():
    source = Path(__file__).resolve().parents[1] / "configs/evaluation/cifar10_edm_posterior.yaml"
    raw = yaml.safe_load(resolved_config_bytes(
        source, ["posterior_t_values=[0.5, 0.5000001, 0.25]"], category="evaluate",
    ))
    times = metric_evaluation_config_from_mapping(raw).posterior_t_values
    assert times == (0.5, 0.5000001, 0.25)

    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.noise = []

        def forward(self, x, t, labels):
            # The held-out endpoint is zero, so recover the source noise.
            self.noise.append((x / (1 - t[:, None, None, None])).clone())
            return {"delta": torch.zeros_like(x), "logvar": torch.zeros_like(x)}

    model = RecordingModel()
    arguments = dict(
        model=model, variant="pfm_delta", config=ExperimentConfig(image_channels=1, image_size=2),
        data_loader=[(torch.zeros(4, 1, 2, 2), torch.zeros(4, dtype=torch.long))],
        device=torch.device("cpu"), num_samples=4, num_bins=2, seed=321,
    )
    forward = evaluate_controlled_interpolant_posterior(**arguments, t_values=times)
    reverse = evaluate_controlled_interpolant_posterior(**arguments, t_values=times[::-1])
    subset = evaluate_controlled_interpolant_posterior(**arguments, t_values=(times[0],))
    assert forward["t_values"] == list(times)
    assert list(forward["by_t"]) == ["0.5", "0.5000001", "0.25"]
    assert [row["t"] for row in forward["by_t"].values()] == list(times)
    assert forward["by_t"] == reverse["by_t"]
    assert forward["by_t"]["0.5"] == subset["by_t"]["0.5"]
    assert not torch.allclose(model.noise[0], model.noise[1])


@pytest.mark.parametrize("times", [
    [], [True], ["0.5"], [float("nan")], [float("inf")],
    [0.0], [1.0], [-0.1], [0.99999999], [1e-50],
    [0.5, 0.5], [0.5, 0.50000001],
])
def test_invalid_times_fail_in_cli_and_direct_diagnostics_before_forward(times):
    source = Path(__file__).resolve().parents[1] / "configs/evaluation/cifar10_edm_posterior.yaml"
    assignment = "posterior_t_values=" + yaml.safe_dump(times, default_flow_style=True).strip()
    with pytest.raises(ValueError, match="posterior_t_values"):
        resolved_config_bytes(source, [assignment], category="evaluate")
    with pytest.raises(ValueError, match="posterior_t_values"):
        evaluate_controlled_interpolant_posterior(
            torch.nn.Identity(), "pfm_delta", ExperimentConfig(), [], torch.device("cpu"),
            num_samples=2, t_values=times,
        )


def test_controlled_diagnostics_refuses_partial_time_coverage(monkeypatch):
    monkeypatch.setattr(posterior_diagnostics_module, "posterior_time_key", lambda _: "collision")

    class Posterior(torch.nn.Module):
        def forward(self, x, t, labels):
            return {"delta": torch.zeros_like(x), "logvar": torch.zeros_like(x)}

    with pytest.raises(ValueError, match="cover every requested time exactly once"):
        evaluate_controlled_interpolant_posterior(
            Posterior(), "pfm_delta", ExperimentConfig(image_channels=1, image_size=2),
            [(torch.zeros(2, 1, 2, 2), torch.zeros(2, dtype=torch.long))],
            torch.device("cpu"), num_samples=2, num_bins=2, t_values=(0.25, 0.5),
        )


def _standardized_residual(component_count: int) -> dict[str, object]:
    accumulator = StandardizedResidualAccumulator(include_time=False)
    accumulator.update(torch.ones(1, component_count), torch.tensor(0.5))
    return accumulator.summary(num_bins=1, t_min=0.0, t_max=1.0)


def _raw_logvar_saturation(component_count: int) -> dict[str, object]:
    return {
        "definition": "sigmoid_raw_tail_probability",
        "tail_probability": 0.01,
        "lower_raw_threshold": -4.59511985013459,
        "upper_raw_threshold": 4.59511985013459,
        "component_count": component_count,
        "lower_count": 0,
        "upper_count": 0,
        "lower_rate": 0.0,
        "upper_rate": 0.0,
    }


def test_controlled_posterior_uses_fixed_times_and_training_noise_scale(monkeypatch):
    config = ExperimentConfig(image_channels=1, image_size=2, noise_scale=3.0)
    seen_xt: list[torch.Tensor] = []
    seen_t: list[torch.Tensor] = []

    class RecordingDeltaModel(torch.nn.Module):
        def forward(self, x, t, labels):
            del labels
            seen_xt.append(x.detach().clone())
            seen_t.append(t.detach().clone())
            return {"delta": torch.zeros_like(x), "logvar": torch.zeros_like(x)}

    monkeypatch.setattr(
        posterior_residuals_module,
        "_randn_like",
        lambda x, generator: torch.ones_like(x),
    )
    loader = [(torch.zeros(2, 1, 2, 2), torch.zeros(2, dtype=torch.long))]
    result = evaluate_controlled_interpolant_posterior(
        RecordingDeltaModel(),
        "pfm_delta",
        config,
        loader,
        torch.device("cpu"),
        t_values=(0.25, 0.5),
        num_samples=2,
        num_bins=2,
    )

    assert result["t_values"] == [0.25, 0.5]
    assert torch.allclose(seen_t[0], torch.full((2,), 0.25))
    assert torch.allclose(seen_t[1], torch.full((2,), 0.5))
    assert torch.allclose(seen_xt[0], torch.full_like(seen_xt[0], 2.25))
    assert torch.allclose(seen_xt[1], torch.full_like(seen_xt[1], 1.5))
    assert result["by_t"]["0.5"]["endpoint"]["mean_pred_var"] == pytest.approx(0.25)


def test_controlled_posterior_metadata_comes_from_residual_statistics(monkeypatch):
    config = ExperimentConfig(image_channels=1, image_size=2)

    def fake_collect_posterior_residual_statistics(**kwargs):
        fixed_t = kwargs["t_sampler"](
            config,
            2,
            torch.device("cpu"),
            "pfm_delta",
            torch.float32,
        )
        assert fixed_t.shape == (2, 1, 1, 1)
        summary = posterior_residuals_module.summarize_variance_calibration(
            torch.ones(2),
            torch.ones(2),
            num_bins=2,
        )
        summary["component_count"] = 8
        return {
            "num_batches": 1,
            "num_samples": 2,
            "actual_nfe_per_batch": 1,
            "standardized_residual": _standardized_residual(8),
            "variance_calibration": {
                "velocity": deepcopy(summary),
                "endpoint": deepcopy(summary),
            },
            "variance_nll": {
                key: 1.0
                for key in posterior_residuals_module.NLL_TENSOR_KEYS
            },
            "raw_logvar_saturation": _raw_logvar_saturation(8),
        }

    monkeypatch.setattr(
        posterior_diagnostics_module,
        "collect_posterior_residual_statistics",
        fake_collect_posterior_residual_statistics,
    )
    result = evaluate_controlled_interpolant_posterior(
        torch.nn.Identity(),
        "pfm_delta",
        config,
        data_loader=[],
        device=torch.device("cpu"),
        t_values=(0.1, 0.5),
        num_samples=2,
        num_bins=2,
    )

    assert result["by_t"]["0.1"]["standardized_residual"] == _standardized_residual(8)
    assert "self_conditioning" not in result
    assert result["actual_nfe_per_batch"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("num_batches"), "num_batches is required"),
        (lambda payload: payload.__setitem__("num_samples", 2.0), "num_samples must be an integer"),
        (
            lambda payload: payload.__setitem__("actual_nfe_per_batch", 2),
            "actual_nfe_per_batch does not match current forward policy",
        ),
        (lambda payload: payload.pop("standardized_residual"), "standardized_residual is required"),
        (
            lambda payload: payload["standardized_residual"].__setitem__("mean", float("nan")),
            "mean must be finite",
        ),
        (
            lambda payload: payload["standardized_residual"].__setitem__("gaussian_coverage_95", 0.5),
            "coverage does not match its count",
        ),
        (
            lambda payload: payload.__setitem__("variance_calibration", {}),
            "variance_calibration must be a non-empty mapping",
        ),
        (
            lambda payload: payload["variance_calibration"].__setitem__("velocity", {}),
            "velocity must be a non-empty mapping",
        ),
        (
            lambda payload: payload["variance_calibration"]["velocity"].__setitem__(
                "component_count", 3
            ),
            "component_count does not match current image shape",
        ),
        (
            lambda payload: payload["variance_calibration"]["velocity"].pop(
                "histogram_bins"
            ),
            "variance summary must contain exactly the current fields",
        ),
        (lambda payload: payload.pop("variance_nll"), "variance_nll is required"),
        (
            lambda payload: payload.pop("raw_logvar_saturation"),
            "raw_logvar_saturation is required",
        ),
        (
            lambda payload: payload["raw_logvar_saturation"].__setitem__(
                "component_count", 7
            ),
            "raw_logvar_saturation component_count does not match current image shape",
        ),
        (
            lambda payload: payload["variance_nll"].pop("velocity_complete_nll_per_dim"),
            "variance_nll must contain",
        ),
        (
            lambda payload: payload["variance_nll"].__setitem__(
                "velocity_complete_nll_per_dim", float("nan")
            ),
            "velocity_complete_nll_per_dim must be finite",
        ),
    ],
)
def test_controlled_posterior_rejects_missing_or_malformed_current_producer_shape(
    monkeypatch,
    mutation,
    message,
):
    config = ExperimentConfig(image_channels=1, image_size=2)
    summary = posterior_residuals_module.summarize_variance_calibration(
        torch.ones(2),
        torch.ones(2),
        num_bins=2,
    )
    summary["component_count"] = 8
    payload = {
        "num_batches": 1,
        "num_samples": 2,
        "actual_nfe_per_batch": 1,
        "standardized_residual": _standardized_residual(8),
        "variance_calibration": {
            "velocity": deepcopy(summary),
            "endpoint": deepcopy(summary),
        },
        "variance_nll": {
            key: 1.0
            for key in posterior_residuals_module.NLL_TENSOR_KEYS
        },
        "raw_logvar_saturation": _raw_logvar_saturation(8),
    }
    mutation(payload)
    monkeypatch.setattr(
        posterior_diagnostics_module,
        "collect_posterior_residual_statistics",
        lambda **kwargs: payload,
    )

    with pytest.raises(ValueError, match=message):
        evaluate_controlled_interpolant_posterior(
            torch.nn.Identity(),
            "pfm_delta",
            config,
            data_loader=[],
            device=torch.device("cpu"),
            t_values=(0.5,),
            num_samples=2,
            num_bins=2,
        )
