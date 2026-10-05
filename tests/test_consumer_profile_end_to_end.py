from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pfm.core.sampling_config import SamplingConfig

from pfm.core.config import ExperimentConfig, experiment_config_to_mapping
from pfm.core.evaluation_config import (
    GeneratedCommonEvaluationConfig,
    PosteriorDiagnosticsEvaluationConfig,
)
from pfm.core.execution import RuntimePaths
from pfm.evaluation import metric_evaluate
from pfm.evaluation.checkpoint_loading import BoundEpochCheckpoint


def _bound(tmp_path: Path) -> BoundEpochCheckpoint:
    checkpoint = tmp_path / "checkpoint.pt"
    manifest = tmp_path / "checkpoint_manifest.json"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text("{}\n", encoding="utf-8")
    return BoundEpochCheckpoint(
        epoch_id="epoch_0001",
        epoch_dir=tmp_path,
        checkpoint_path=checkpoint,
        manifest_path=manifest,
        manifest={},
        config_identity={},
        checkpoint_runtime_identity={
            "project_name": "pfm",
            "git_commit": "a" * 40,
            "rank": 0,
            "world_size": 1,
        },
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )


def _runtime(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths(
        project_name="pfm",
        project_root=tmp_path / "project",
        datasets_root=tmp_path / "datasets",
        assets_root=tmp_path / "scratch" / "pfm" / "assets",
        runs_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
    )


def test_generated_common_profile_reaches_streaming_service_without_legacy_fields(
    monkeypatch,
    tmp_path: Path,
) -> None:
    training = ExperimentConfig(dataset="mnist", num_classes=10)
    checkpoint = {
        "training_config": experiment_config_to_mapping(training),
        "variant": "velocity_fm",
    }
    config = GeneratedCommonEvaluationConfig(
        profile="generated_common",
        metric_names=("conditional_accuracy",),
        generated_population=10,
        sampling=SamplingConfig(steps=2),
        batch_size=5,
        fid_batch_size=5,
        fid_backend="inception",
        seed=321,
        classifier_asset_id="mnist-feature-seed123-bf16-pfm-v4",
        fid_asset_id=None,
    )
    calls = {}
    checkpoint_loads = []
    model_builds = []

    def fake_checkpoint_load(*args, **kwargs):
        checkpoint_loads.append((args, kwargs))
        return checkpoint

    monkeypatch.setattr(metric_evaluate, "load_bound_training_checkpoint", fake_checkpoint_load)
    monkeypatch.setattr(
        metric_evaluate,
        "_build_model_from_metric_checkpoint",
        lambda *_, **__: model_builds.append(True) or torch.nn.Identity(),
    )
    monkeypatch.setattr(
        metric_evaluate,
        "validate_evaluation_asset_bindings",
        lambda *_: SimpleNamespace(dataset="mnist", image_size=28, fid_backend="inception"),
    )
    monkeypatch.setattr(
        metric_evaluate,
        "load_classifier_asset",
        lambda *_: (
            torch.nn.Identity(),
            {
                "dataset": "mnist",
                "image_size": 28,
                "num_classes": 10,
                "config": {"image_channels": 1},
            },
        ),
    )

    def fake_generation(**kwargs):
        calls.update(kwargs)
        kwargs["load_model"]()
        return {
            "generated_protocol": {
                "generation_count": 10,
                "sampling": SamplingConfig(steps=2).to_dict(),
                "actual_nfe": 2,
                "actual_vjp": 0,
            },
            "metrics": {"conditional_accuracy": {"accuracy": 1.0}},
        }

    monkeypatch.setattr(metric_evaluate, "evaluate_generation", fake_generation)
    result = metric_evaluate.evaluate_checkpoint_metrics(
        _bound(tmp_path),
        tmp_path / "artifact",
        _runtime(tmp_path),
        config,
        torch.device("cpu"),
    )

    assert calls["generated_population"] == 10
    assert calls["sampling"] == SamplingConfig(steps=2)
    assert "fid_stats" not in calls
    assert "output_dir" not in calls
    assert len(checkpoint_loads) == 1
    assert checkpoint_loads[0][1] == {}
    assert model_builds == [True]
    execution = result["aggregate"]["generated_protocol"]
    assert execution == {
        "generation_count": 10,
        "sampling": SamplingConfig(steps=2).to_dict(),
        "actual_nfe": 2,
        "actual_vjp": 0,
    }


def test_posterior_profile_passes_explicit_sample_count_to_both_independent_suites(
    monkeypatch,
    tmp_path: Path,
) -> None:
    training = ExperimentConfig(dataset="mnist", num_classes=10)
    checkpoint = {
        "training_config": experiment_config_to_mapping(training),
        "variant": "pfm_full",
    }
    config = PosteriorDiagnosticsEvaluationConfig(
        profile="posterior_diagnostics",
        metric_names=("posterior_controlled", "posterior_calibration"),
        posterior_population=4,
        batch_size=2,
        num_bins=15,
        posterior_t_values=(0.5,),
        seed=321,
    )
    calls = {}
    monkeypatch.setattr(
        metric_evaluate,
        "load_bound_training_checkpoint",
        lambda *_, **__: checkpoint,
    )
    monkeypatch.setattr(
        metric_evaluate,
        "validate_evaluation_asset_bindings",
        lambda *_: SimpleNamespace(dataset="mnist", image_size=28, fid_backend="inception"),
    )
    def build_posterior(*_args, execution):
        calls["execution"] = execution
        return torch.nn.Identity()

    def loaders(config, *_):
        calls["loader_config"] = config
        return [], []

    monkeypatch.setattr(metric_evaluate, "_build_model_from_metric_checkpoint", build_posterior)
    monkeypatch.setattr(metric_evaluate, "make_dataloaders", loaders)

    space = {
        "count": 4,
        "mean_pred_var": 1.0,
        "mean_squared_error": 1.0,
        "error_to_var_ratio": 1.0,
        "bins": [
            {
                "count": 4,
                "mean_pred_var": 1.0,
                "mean_squared_error": 1.0,
                "error_to_var_ratio": 1.0,
            }
        ],
        "histogram_bins": [
            {"count": 4, "center_pred_var": 1.0, "fraction": 1.0}
        ],
    }
    controlled = {
        "num_samples": 4,
        "by_t": {
            "0.5": {
                "t": 0.5,
                "velocity": space,
                "endpoint": space,
                "variance_nll": {
                    "velocity_complete_nll_per_dim": 1.0,
                    "endpoint_complete_nll_per_dim": 0.5,
                },
            }
        }
    }
    random = {
        "num_samples": 4,
        "variance_calibration": {"velocity": space, "endpoint": space},
        "time_calibration": {
            "velocity": {
                "count": 4,
                "bins": [{"t_mean": 0.5, **space["bins"][0]}],
            },
            "endpoint": {
                "count": 4,
                "bins": [{"t_mean": 0.5, **space["bins"][0]}],
            },
        },
        "variance_nll": {},
    }

    def fake_controlled(*_args, **kwargs):
        calls["controlled"] = kwargs
        return controlled

    def fake_random(*_args, **kwargs):
        calls["random"] = kwargs
        return random

    monkeypatch.setattr(metric_evaluate, "evaluate_controlled_interpolant_posterior", fake_controlled)
    monkeypatch.setattr(metric_evaluate, "evaluate_pfm_calibration", fake_random)
    monkeypatch.setattr(
        metric_evaluate,
        "write_posterior_figures",
        lambda *_: {name: {} for name in (
            "controlled_variance_error",
            "controlled_complete_nll",
            "random_variance_calibration",
            "random_error_variance_ratio",
            "random_variance_histogram",
            "random_time_calibration",
        )},
    )
    result = metric_evaluate.evaluate_checkpoint_metrics(
        _bound(tmp_path),
        tmp_path / "artifact",
        _runtime(tmp_path),
        config,
        torch.device("cpu"),
    )

    assert calls["execution"] == config.execution
    assert calls["loader_config"].mixed_precision == "none"
    assert calls["loader_config"].jit_compile is False
    assert calls["loader_config"].num_workers == config.num_workers
    assert training.mixed_precision == "bf16"
    assert calls["controlled"]["num_samples"] == 4
    assert calls["random"]["num_samples"] == 4
    assert "num_batches" not in calls["controlled"]
    assert "num_batches" not in calls["random"]
    assert "sample_count" not in result["aggregate"]["controlled"][0]
    assert set(result["presentation"]) == {
        "controlled_variance_error",
        "controlled_complete_nll",
        "random_variance_calibration",
        "random_error_variance_ratio",
        "random_variance_histogram",
        "random_time_calibration",
    }

    controlled["num_samples"] = 3
    with pytest.raises(ValueError, match="controlled posterior population"):
        metric_evaluate.evaluate_checkpoint_metrics(
            _bound(tmp_path),
            tmp_path / "invalid-artifact",
            _runtime(tmp_path),
            config,
            torch.device("cpu"),
        )
    controlled["num_samples"] = 4
    random["num_samples"] = 3
    with pytest.raises(ValueError, match="random-time posterior population"):
        metric_evaluate.evaluate_checkpoint_metrics(
            _bound(tmp_path),
            tmp_path / "invalid-random-artifact",
            _runtime(tmp_path),
            config,
            torch.device("cpu"),
        )
