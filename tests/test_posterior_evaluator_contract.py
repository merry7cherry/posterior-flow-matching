import json
import math

import pytest
import torch

from pfm.core.config import ExperimentConfig
from pfm.core.serialization import to_jsonable
from pfm.diagnostics.calibration_metrics import evaluate_pfm_calibration
from pfm.diagnostics.posterior_diagnostics import evaluate_controlled_interpolant_posterior
from pfm.diagnostics.posterior_residuals import summarize_variance_calibration
from pfm.diagnostics.posterior_residuals import summarize_time_calibration


def test_variance_calibration_preserves_ties_and_is_permutation_invariant():
    predicted_variance = torch.tensor([1.0, 1.0, 1.0, 9.0])
    squared_error = torch.tensor([0.0, 1.0, 2.0, 3.0])
    histogram_edges = torch.tensor([0.5, 2.0, 10.0])

    expected = summarize_variance_calibration(
        predicted_variance,
        squared_error,
        num_bins=2,
        histogram_edges=histogram_edges,
    )
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = summarize_variance_calibration(
        predicted_variance[permutation],
        squared_error[permutation],
        num_bins=2,
        histogram_edges=histogram_edges,
    )

    assert expected == permuted
    assert expected["requested_num_bins"] == 2
    assert expected["num_bins"] == 2
    assert [item["count"] for item in expected["bins"]] == [3, 1]
    assert expected["spearman_var_error"] == pytest.approx(math.sqrt(3.0 / 5.0))
    assert expected["spearman_var_error_reason"] is None
    assert expected["histogram_edges"] == [0.5, 2.0, 10.0]


def test_undefined_rank_statistic_is_explicit_and_strict_json():
    summary = summarize_variance_calibration(
        torch.ones(4),
        torch.arange(4, dtype=torch.float32),
        num_bins=2,
        histogram_edges=torch.tensor([0.5, 1.0, 2.0]),
    )

    assert summary["spearman_var_error"] is None
    assert summary["spearman_var_error_reason"] == "constant_rank"
    json.dumps(to_jsonable(summary), allow_nan=False)


def test_random_time_bins_use_the_common_bounded_interval():
    times = torch.tensor([0.1, 0.299, 0.31, 0.51, 0.71, 0.9]).view(-1, 1, 1, 1)
    variance = torch.ones_like(times)
    error = torch.ones_like(times)

    summary = summarize_time_calibration(
        times,
        variance,
        error,
        num_bins=4,
        t_min=0.1,
        t_max=0.9,
    )

    assert summary["time_binning"] == "equal_width_bounded_interval"
    assert summary["time_bounds"] == [0.1, 0.9]
    assert [item["count"] for item in summary["bins"]] == [2, 1, 1, 2]

    with pytest.raises(ValueError, match="outside the authored bounded interval"):
        summarize_time_calibration(
            torch.tensor([0.05]),
            torch.ones(1),
            torch.ones(1),
            t_min=0.1,
            t_max=0.9,
        )


def test_controlled_order_and_random_suite_are_independently_reproducible():
    config = ExperimentConfig(
        image_channels=1,
        image_size=2,
        logvar_min=-1.0,
        logvar_max=1.0,
        eps_t=0.1,
        time_sampling="uniform",
    )
    images = torch.tensor(
        [
            [[[0.0, 0.5], [1.0, 1.5]]],
            [[[1.5, 1.0], [0.5, 0.0]]],
        ]
    )
    data_loader = [(images, torch.zeros(images.shape[0], dtype=torch.long))]

    class ZeroDeltaModel(torch.nn.Module):
        def forward(self, x, t, labels):
            del t, labels
            return {
                "delta": torch.zeros_like(x),
                "logvar": torch.zeros_like(x),
            }

    model = ZeroDeltaModel()
    controlled_forward = evaluate_controlled_interpolant_posterior(
        model,
        "pfm_delta",
        config,
        data_loader,
        torch.device("cpu"),
        t_values=(0.25, 0.75),
        num_samples=2,
        num_bins=2,
        seed=321,
    )
    random_alone = evaluate_pfm_calibration(
        model,
        "pfm_delta",
        config,
        data_loader,
        torch.device("cpu"),
        num_samples=2,
        num_bins=2,
        seed=321,
    )

    torch.manual_seed(999)
    torch.randn(128)
    controlled_reverse = evaluate_controlled_interpolant_posterior(
        model,
        "pfm_delta",
        config,
        data_loader,
        torch.device("cpu"),
        t_values=(0.75, 0.25),
        num_samples=2,
        num_bins=2,
        seed=321,
    )
    random_after_controlled = evaluate_pfm_calibration(
        model,
        "pfm_delta",
        config,
        data_loader,
        torch.device("cpu"),
        num_samples=2,
        num_bins=2,
        seed=321,
    )

    assert controlled_forward["by_t"] == controlled_reverse["by_t"]
    assert random_alone == random_after_controlled


def test_method_profiles_own_distinct_posterior_rng_streams():
    images = torch.zeros(2, 1, 2, 2)
    data_loader = [(images, torch.zeros(images.shape[0], dtype=torch.long))]

    class ZeroDeltaModel(torch.nn.Module):
        def forward(self, x, t, labels):
            del t, labels
            return {
                "mu": x,
                "delta": torch.zeros_like(x),
                "logvar": torch.zeros_like(x),
            }

    results = {}
    for variant in ("pfm_full", "pfm_delta"):
        config = ExperimentConfig(
            image_channels=1,
            image_size=2,
        )
        result = evaluate_controlled_interpolant_posterior(
            ZeroDeltaModel(),
            variant,
            config,
            data_loader,
            torch.device("cpu"),
            t_values=(0.5,),
            num_samples=2,
            num_bins=2,
            seed=321,
        )
        results[variant] = result

    assert {
        variant: result["rng"]["profile"] for variant, result in results.items()
    } == {
        "pfm_delta": "pfm_delta",
        "pfm_full": "pfm_full",
    }
    errors = {
        next(iter(result["by_t"].values()))["velocity"]["mean_squared_error"]
        for result in results.values()
    }
    assert len(errors) == 2
