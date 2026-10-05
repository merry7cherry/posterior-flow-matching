import itertools
import math

import pytest
import torch
import yaml
from dataclasses import replace
from pathlib import Path

from pfm.core.sde_acceptance_config import load_sde_acceptance_config

from pfm.cli.sde_acceptance import parse_args
from pfm.evaluation.sde_acceptance import (
    coupled_brownian_driver,
    coupled_step_random,
    independent_group_error,
)
from pfm.sampling import samplers
from pfm.sampling.randomness import indexed_step_random


def test_coupled_driver_coarse_increment_is_sum_of_fine_increments():
    indices = torch.tensor([0, 63, 64])
    arguments = dict(seed=42, stream="brownian", device=torch.device("cpu"))
    coarse = coupled_step_random(indices, (2,), step=3, refinement=2, **arguments)
    fine_a = coupled_step_random(indices, (2,), step=6, refinement=1, **arguments)
    fine_b = coupled_step_random(indices, (2,), step=7, refinement=1, **arguments)
    h = 0.98 / 100
    torch.testing.assert_close(math.sqrt(2 * h) * coarse, math.sqrt(h) * (fine_a + fine_b))


def test_coupled_driver_aligns_probe_time_indices_and_restores_production_function():
    original = samplers.indexed_step_random
    arguments = dict(seed=42, stream="derivative", probe=3, device=torch.device("cpu"))
    indices = torch.tensor([0])
    with coupled_brownian_driver(50, finest=100):
        actual = samplers.indexed_step_random(indices, (4,), step=2, **arguments)
    expected = indexed_step_random(indices, (4,), step=4, **arguments)
    assert samplers.indexed_step_random is original
    torch.testing.assert_close(actual, expected)
    with pytest.raises(RuntimeError):
        with coupled_brownian_driver(50, finest=100):
            raise RuntimeError("abort bounded validation")
    assert samplers.indexed_step_random is original


def test_independent_group_error_matches_exact_diagonal_mse_on_all_sign_pairs():
    jacobian = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    estimates = []
    for signs in itertools.product((-1.0, 1.0), repeat=2):
        z = torch.tensor(signs, dtype=torch.float64)
        estimates.append(z * (jacobian.T @ z))
    exact_mse = sum(float((estimate - jacobian.diag()).square().mean()) for estimate in estimates) / len(estimates)
    paired_mse = sum(independent_group_error(first, second) for first in estimates for second in estimates) / len(estimates) ** 2
    assert exact_mse == pytest.approx((2**2 + 3**2) / 2)
    assert paired_mse == pytest.approx(exact_mse)


@pytest.mark.parametrize("field,value", [
    ("batch_size", 17), ("probe_repeats", 0), ("seed", -1),
    ("diffusion_scale", float("nan")), ("diffusion_scale", 0),
    ("terminal_time", 1), ("compile", "false"), ("precisions", ["fp16"]),
    ("sampler_steps", [100, 250]), ("probe_counts", [2, 2]),
    ("reference_times", [0.999]), ("cutoff_times", [0.98, 1]),
])
def test_invalid_bounded_benchmark_protocol_fails_during_config_preflight(tmp_path, field, value):
    raw = yaml.safe_load(Path("configs/diagnostics/cifar10_sde_acceptance.yaml").read_text())
    raw[field] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises((ValueError, TypeError)):
        load_sde_acceptance_config(path)


def test_acceptance_parser_defaults_match_bounded_campaign():
    args = parse_args(["--variant-root", "/unused", "--epoch", "600", "--diagnostic-id", "acceptance", "--config", "configs/diagnostics/cifar10_sde_acceptance.yaml"])
    config = load_sde_acceptance_config(args.config)
    assert (config.batch_size, config.probe_repeats, config.terminal_time, config.diffusion_scale) == (4, 2, 0.99, 0.1)




@pytest.mark.parametrize("variant", ["pfm_full", "pfm_delta"])
def test_generic_benchmark_reuses_sampler_and_labels_coupling_correctly(monkeypatch, variant):
    from pfm.evaluation import sde_acceptance as module
    from pfm.core.config import ExperimentConfig

    class Posterior(torch.nn.Module):
        def forward(self, x, t, labels):
            velocity = 0.05 * x
            return {"delta": velocity, "mu": x + (1-t) * velocity, "logvar": 0.2 * x}

    protocol = replace(load_sde_acceptance_config("configs/diagnostics/imagenet256_sde_acceptance.yaml"),
                       precisions=("fp32",), sampler_steps=(2, 4), terminal_time=0.8, cutoff_times=(0.8, 0.9),
                       reference_times=(0.1, 0.5))
    monkeypatch.setattr(module, "_cuda_measure", lambda _: 0)
    monkeypatch.setattr(module, "_cuda_result", lambda *_: {"elapsed_seconds": 0, "peak_memory_allocated_bytes": 0})
    records, comparisons, samples = module.sampler_benchmark(
        Posterior(), variant,
        ExperimentConfig(image_channels=1, image_size=2, num_classes=2, logvar_min=-1, logvar_max=1),
        torch.tensor([0, 1]), protocol, torch.device("cpu"))
    assert len(records) == 12 and len(samples) == 12
    assert {c["kind"] for c in comparisons} == {
        "cutoff_common_normalized_random_inputs_not_same_physical_brownian_path",
        "step_refinement_coupled_brownian_path", "ode_step_refinement", "probe_budget_common_brownian_path",
    }
    assert all(r["numerical_summary"]["nonfinite_count"] == 0 for r in records)
    for r in records:
        assert r["actual_nfe"] == r["sampling"]["steps"] + 1
        assert r["actual_vjp"] == (r["sampling"]["steps"] * r["sampling"]["derivative_probes"] if r["sampling"]["solver"] == "sde" else 0)


def test_probe_benchmark_accepts_single_precision_without_fake_comparison(monkeypatch):
    from pfm.evaluation import sde_acceptance as module
    from pfm.core.config import ExperimentConfig

    class Posterior(torch.nn.Module):
        def forward(self, x, t, labels):
            return {"mu": x * .1, "logvar": .2 * x}

    protocol = replace(load_sde_acceptance_config("configs/diagnostics/imagenet256_sde_acceptance.yaml"),
                       precisions=("fp32",), probe_counts=(1, 2), sampler_steps=(2, 4), reference_times=(.1, .5))
    monkeypatch.setattr(module, "_cuda_measure", lambda _: 0)
    monkeypatch.setattr(module, "_cuda_result", lambda *_: {"elapsed_seconds": 0, "peak_memory_allocated_bytes": 0})
    x = torch.ones(2, 1, 2, 2)
    records, comparisons = module.probe_benchmark(
        Posterior(), "pfm_full", ExperimentConfig(logvar_min=-1, logvar_max=1),
        x, -x, torch.tensor([0, 1]), protocol, torch.device("cpu"))
    assert len(records) == 4 and comparisons == []
    assert all(r["actual_vjp"] == 2 * protocol.probe_repeats * r["derivative_probes"] for r in records)
    assert all(r["derivative_mse_estimate"] == pytest.approx(0, abs=1e-12) for r in records)
