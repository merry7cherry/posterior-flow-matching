from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pfm.core.sampling_config import SamplingConfig

from pfm.core.evaluation_config import PosteriorDiagnosticsEvaluationConfig
from pfm.core.sample_config import SampleVisualizationConfig
from pfm.data.identity import DATASET_MANIFEST_NAME
from pfm.evaluation import checkpoint_evaluation, checkpoint_loading, sample_grid
from pfm.evaluation.checkpoint_loading import BoundEpochCheckpoint


def _bound(variant_root: Path) -> BoundEpochCheckpoint:
    epoch_dir = variant_root / "epochs" / "epoch_0001"
    runtime_identity = {
        "project_name": "pfm",
        "git_commit": "a" * 40,
        "rank": 0,
        "world_size": 1,
    }
    config_identity = {
        "dataset": "mnist",
        "architecture": "conditional_unet",
        "variant": "pfm_full",
        "runtime_identity": runtime_identity,
        "dataset_scientific_identity_sha256": "b" * 64,
        "shared_training_protocol_sha256": "c" * 64,
    }
    return BoundEpochCheckpoint(
        epoch_id="epoch_0001",
        epoch_dir=epoch_dir,
        checkpoint_path=epoch_dir / "checkpoint.pt",
        manifest_path=epoch_dir / "checkpoint_manifest.json",
        manifest={
            "variant_id": "pfm_full",
            "epoch": 1,
            "step": 100,
            "config_identity": config_identity,
        },
        config_identity=config_identity,
        checkpoint_runtime_identity=runtime_identity,
        checkpoint_sha256="d" * 64,
    )


def test_evaluation_dataset_mismatch_fails_before_metric_or_model_work(
    monkeypatch,
    tmp_path: Path,
    execution_context,
) -> None:
    variant_root = execution_context.runtime.experiments_root / "task" / "pfm_full"
    config_path = execution_context.runtime.project_root / "posterior.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("profile: posterior_diagnostics\n", encoding="utf-8")
    bound = _bound(variant_root)
    monkeypatch.setattr(checkpoint_evaluation, "resolve_bound_epoch_checkpoint", lambda *_: bound)
    monkeypatch.setattr(
        checkpoint_loading,
        "validate_bound_checkpoint_dataset",
        lambda *_: (_ for _ in ()).throw(ValueError("dataset mismatch")),
    )
    monkeypatch.setattr(
        checkpoint_evaluation,
        "evaluate_checkpoint_metrics",
        lambda **_: (_ for _ in ()).throw(AssertionError("expensive metric/model work ran")),
    )
    config = PosteriorDiagnosticsEvaluationConfig(
        profile="posterior_diagnostics",
        metric_names=("posterior_controlled", "posterior_calibration"),
        posterior_population=2,
        batch_size=2,
        num_bins=15,
        posterior_t_values=(0.5,),
        seed=321,
    )
    with pytest.raises(ValueError, match="dataset mismatch"):
        checkpoint_evaluation.evaluate_bound_checkpoint(
            variant_root,
            1,
            "posterior",
            config,
            evaluation_config_path=config_path,
            job_context=execution_context,
            device=torch.device("cpu"),
        )


def test_sample_dataset_mismatch_fails_before_checkpoint_model_load(
    monkeypatch,
    tmp_path: Path,
    execution_context,
) -> None:
    variant_root = execution_context.runtime.experiments_root / "task" / "pfm_full"
    config_path = execution_context.runtime.project_root / "sample.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("profile: sample_visualization\n", encoding="utf-8")
    bound = _bound(variant_root)
    monkeypatch.setattr(sample_grid, "resolve_bound_epoch_checkpoint", lambda *_: bound)
    monkeypatch.setattr(
        checkpoint_loading,
        "validate_bound_checkpoint_dataset",
        lambda *_: (_ for _ in ()).throw(ValueError("dataset mismatch")),
    )
    monkeypatch.setattr(
        sample_grid,
        "load_model_from_bound_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("checkpoint model state loaded")
        ),
    )
    config = SampleVisualizationConfig(
        profile="sample_visualization",
        sampling=SamplingConfig(steps=50),
        batch_size=8,
        class_start=0,
        class_count=10,
        num_per_class=8,
        seed=321,
        uncertainty_capture_t=0.5,

    )
    with pytest.raises(ValueError, match="dataset mismatch"):
        sample_grid.generate_sample_artifact(
            variant_root,
            1,
            "sample",
            config,
            sample_config_path=config_path,
            job_context=execution_context,
            device=torch.device("cpu"),
        )


def test_sample_accepts_project_alias_and_binds_current_consumer_revision(
    monkeypatch,
    tmp_path: Path,
    execution_context,
) -> None:
    physical_project_root = execution_context.runtime.project_root
    physical_project_root.mkdir(parents=True)
    logical_project_root = tmp_path / "logical-project"
    logical_project_root.symlink_to(physical_project_root, target_is_directory=True)
    runtime = replace(execution_context.runtime, project_root=logical_project_root)
    job_context = replace(execution_context, runtime=runtime)
    logical_config_path = logical_project_root / "sample.yaml"
    logical_config_path.write_text("profile: sample_visualization\n", encoding="utf-8")
    config_path = runtime.project_file("sample.yaml", name="sample config")

    assert config_path == (physical_project_root / "sample.yaml").resolve()
    variant_root = job_context.runtime.experiments_root / "task" / "pfm_full"
    bound = _bound(variant_root)
    bound.checkpoint_runtime_identity["git_commit"] = "0" * 40
    dataset_identity = job_context.runtime.dataset_root("mnist") / DATASET_MANIFEST_NAME
    dataset_identity.parent.mkdir(parents=True)
    dataset_identity.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        sample_grid,
        "resolve_bound_epoch_checkpoint",
        lambda *_: bound,
    )
    monkeypatch.setattr(
        checkpoint_loading,
        "validate_bound_checkpoint_dataset",
        lambda *_: {
            "scientific_identity": {
                "class_to_idx": {str(index): index for index in range(10)}
            }
        },
    )
    monkeypatch.setattr(
        sample_grid,
        "load_model_from_bound_checkpoint",
        lambda *_args, **_kwargs: (
            object(),
            "pfm_full",
            SimpleNamespace(batch_size=8, dataset="mnist"),
        ),
    )
    monkeypatch.setattr(sample_grid, "_validate_loaded_profile", lambda *_: None)

    def fake_publish(**kwargs):
        captured.update(kwargs["binding"])
        return kwargs["parent"] / kwargs["artifact_id"]

    monkeypatch.setattr(sample_grid, "publish_consumer_artifact", fake_publish)
    config = SampleVisualizationConfig(
        profile="sample_visualization",
        sampling=SamplingConfig(steps=50),
        batch_size=8,
        class_start=0,
        class_count=10,
        num_per_class=8,
        seed=321,
        uncertainty_capture_t=0.5,

    )

    sample_grid.generate_sample_artifact(
        variant_root,
        1,
        "sample",
        config,
        sample_config_path=config_path,
        job_context=job_context,
        device=torch.device("cpu"),
    )

    assert captured["config"]["path"] == "sample.yaml"
    assert captured["consumer_revision"] == job_context.git_commit
    assert captured["consumer_revision"] != bound.checkpoint_runtime_identity["git_commit"]
    assert captured["checkpoint_revision"] == bound.checkpoint_runtime_identity["git_commit"]


@pytest.mark.parametrize("category", ["evaluate", "sample"])
@pytest.mark.parametrize("route", ["frozen_input", "direct_override"])
def test_frozen_config_reaches_real_consumer_entrypoint(
    category, route, monkeypatch, tmp_path, execution_context,
):
    """Exercise snapshot -> CLI -> service -> binding/compute boundary.

    Stub CUDA and expensive data/model work, preserving the real consumer
    entrypoint and its frozen-input and path validation.
    """
    from pfm.cli import evaluate, sample
    from pfm.cli.config_inputs import execution_config
    from pfm.core.config import ExperimentConfig

    runtime = execution_context.runtime
    runtime.project_root.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    relative = ("configs/evaluation/mnist_posterior.yaml" if category == "evaluate"
                else "configs/sampling/mnist_visualization.yaml")
    source = runtime.project_root / "consumer.yaml"
    source.write_bytes((root / relative).read_bytes())
    variant_root = runtime.experiments_root / "task" / "pfm_full"
    bound = _bound(variant_root)
    dataset_path = runtime.dataset_root("mnist") / DATASET_MANIFEST_NAME
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text("{}\n")
    monkeypatch.setattr(checkpoint_loading, "validate_bound_checkpoint_dataset", lambda *_: {
        "scientific_identity": {"class_to_idx": {str(i): i for i in range(10)}}
    })
    cli = evaluate if category == "evaluate" else sample
    service = checkpoint_evaluation if category == "evaluate" else sample_grid
    monkeypatch.setattr(cli, "context_from_args", lambda args: execution_context)
    monkeypatch.setattr(cli, "execution_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(service, "resolve_bound_epoch_checkpoint", lambda *_: bound)
    captured = {}

    def publish(**kwargs):
        captured["binding"] = kwargs["binding"]
        kwargs["numerical_result"](tmp_path / "artifact")
        return kwargs["parent"] / kwargs["artifact_id"]

    monkeypatch.setattr(service, "publish_consumer_artifact", publish)
    if category == "evaluate":
        def metrics(**kwargs):
            captured["batch_size"] = kwargs["evaluation_config"].batch_size
            return {"profile": "posterior_diagnostics", "method": "pfm_full"}
        monkeypatch.setattr(service, "evaluate_checkpoint_metrics", metrics)
    else:
        monkeypatch.setattr(service, "load_model_from_bound_checkpoint",
                            lambda *_, **__: (object(), "pfm_full", ExperimentConfig(batch_size=257)))
        def generate(model, variant, training_config, request, **kwargs):
            assert training_config.batch_size == 257
            captured["batch_size"] = request.batch_size
            return iter(())
        monkeypatch.setattr(service, "iter_loaded_generation", generate)
        monkeypatch.setattr(service, "SampleVisualizationWriter",
                            lambda **_: SimpleNamespace(finalize=lambda: {"sampling": {}}))

    if route == "frozen_input":
        config_path = execution_config(runtime, str(source), ["batch_size=3"],
                                       category=category, name="consumer config")
        source.write_text("no longer a valid consumer config\n")
        overrides = []
    else:
        config_path, overrides = source, ["--set", "batch_size=3"]
    arguments = ["--variant-root", str(variant_root), "--epoch", "1",
                 "--evaluation-id" if category == "evaluate" else "--sample-id", "snapshot-test",
                 "--config", str(config_path), *overrides]
    assert cli.main(arguments) == 0

    assert captured["batch_size"] == 3
    reference = captured["binding"]["config"]
    assert reference["owner"] == "runs"
    snapshot = runtime.runs_root / reference["path"]
    assert snapshot.is_relative_to(runtime.runs_root / "inputs")
    assert reference["sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()


def test_baseline_posterior_profile_fails_before_dataset_or_checkpoint_load(
    monkeypatch,
    execution_context,
) -> None:
    variant_root = execution_context.runtime.experiments_root / "task" / "velocity_fm"
    config_path = execution_context.runtime.project_root / "posterior.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("profile: posterior_diagnostics\n", encoding="utf-8")
    bound = _bound(variant_root)
    bound.manifest["variant_id"] = "velocity_fm"
    bound.manifest["config_identity"]["variant"] = "velocity_fm"
    monkeypatch.setattr(checkpoint_evaluation, "resolve_bound_epoch_checkpoint", lambda *_: bound)
    monkeypatch.setattr(
        checkpoint_loading,
        "validate_bound_checkpoint_dataset",
        lambda *_: (_ for _ in ()).throw(AssertionError("dataset preflight ran")),
    )
    monkeypatch.setattr(
        checkpoint_evaluation,
        "evaluate_checkpoint_metrics",
        lambda **_: (_ for _ in ()).throw(AssertionError("checkpoint/model work ran")),
    )
    config = PosteriorDiagnosticsEvaluationConfig(
        profile="posterior_diagnostics",
        metric_names=("posterior_controlled", "posterior_calibration"),
        posterior_population=2,
        batch_size=2,
        num_bins=15,
        posterior_t_values=(0.5,),
        seed=321,
    )

    with pytest.raises(ValueError, match="requires learned variance"):
        checkpoint_evaluation.evaluate_bound_checkpoint(
            variant_root,
            1,
            "posterior",
            config,
            evaluation_config_path=config_path,
            job_context=execution_context,
            device=torch.device("cpu"),
        )


def test_invalid_baseline_sde_fails_before_checkpoint_model_load(
    monkeypatch,
    execution_context,
) -> None:
    variant_root = execution_context.runtime.experiments_root / "task" / "velocity_fm"
    config_path = execution_context.runtime.project_root / "sde.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("profile: sample_visualization\n", encoding="utf-8")
    bound = _bound(variant_root)
    bound.manifest["variant_id"] = "velocity_fm"
    bound.manifest["config_identity"]["dataset"] = "imagenet"
    monkeypatch.setattr(sample_grid, "resolve_bound_epoch_checkpoint", lambda *_: bound)
    monkeypatch.setattr(checkpoint_loading, "validate_bound_checkpoint_dataset", lambda *_: {})
    monkeypatch.setattr(
        sample_grid,
        "load_model_from_bound_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("checkpoint model state loaded")
        ),
    )
    config = SampleVisualizationConfig(
        profile="sample_visualization",
        sampling=SamplingConfig(solver="sde", steps=250, terminal_time=0.98, diffusion_scale=0.1),
        batch_size=8,
        class_start=0,
        class_count=10,
        num_per_class=1,
        seed=321,
        uncertainty_capture_t=0.49,

    )
    with pytest.raises(ValueError, match="SDE sampling requires learned variance"):
        sample_grid.generate_sample_artifact(
            variant_root,
            1,
            "sde",
            config,
            sample_config_path=config_path,
            job_context=execution_context,
            device=torch.device("cpu"),
        )


def test_evaluation_binds_current_consumer_revision_not_old_checkpoint_revision(
    monkeypatch,
    execution_context,
) -> None:
    variant_root = execution_context.runtime.experiments_root / "task" / "pfm_full"
    config_path = execution_context.runtime.project_root / "posterior.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("profile: posterior_diagnostics\n", encoding="utf-8")
    dataset_identity = (
        execution_context.runtime.dataset_root("mnist") / DATASET_MANIFEST_NAME
    )
    dataset_identity.parent.mkdir(parents=True, exist_ok=True)
    dataset_identity.write_text("{}\n", encoding="utf-8")
    bound = _bound(variant_root)
    bound.checkpoint_runtime_identity["git_commit"] = "0" * 40
    captured: dict[str, object] = {}

    monkeypatch.setattr(checkpoint_evaluation, "resolve_bound_epoch_checkpoint", lambda *_: bound)
    monkeypatch.setattr(checkpoint_loading, "validate_bound_checkpoint_dataset", lambda *_: {})
    monkeypatch.setattr(
        checkpoint_evaluation,
        "evaluate_checkpoint_metrics",
        lambda **_: {
            "profile": "posterior_diagnostics",
            "method": "pfm_full",
            "aggregate": {},
        },
    )

    def fake_publish(**kwargs):
        captured.update(kwargs["binding"])
        return kwargs["parent"] / kwargs["artifact_id"]

    monkeypatch.setattr(checkpoint_evaluation, "publish_consumer_artifact", fake_publish)
    config = PosteriorDiagnosticsEvaluationConfig(
        profile="posterior_diagnostics",
        metric_names=("posterior_controlled", "posterior_calibration"),
        posterior_population=2,
        batch_size=2,
        num_bins=15,
        posterior_t_values=(0.5,),
        seed=321,
    )
    checkpoint_evaluation.evaluate_bound_checkpoint(
        variant_root,
        1,
        "posterior",
        config,
        evaluation_config_path=config_path,
        job_context=execution_context,
        device=torch.device("cpu"),
    )

    assert captured["consumer_revision"] == execution_context.git_commit
    assert captured["consumer_revision"] != bound.checkpoint_runtime_identity["git_commit"]
    assert captured["checkpoint_revision"] == bound.checkpoint_runtime_identity["git_commit"]
