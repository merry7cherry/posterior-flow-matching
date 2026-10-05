"""Standalone checkpoints retain byte identity and the shared consumer protocol."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

import pytest
import torch

from pfm.core import checkpointing as checkpoint_io
from pfm.core.config import ExperimentConfig, shared_training_protocol_sha256
from pfm.core.evaluation_config import PosteriorDiagnosticsEvaluationConfig
from pfm.core.rng_state import capture_rng_state
from pfm.core.sample_config import SampleVisualizationConfig
from pfm.core.sampling_config import SamplingConfig
from pfm.data import datasets
from pfm.evaluation import checkpoint_evaluation, checkpoint_loading, sample_grid
from pfm.models.factory import build_model
from pfm.training.checkpointing import build_training_checkpoint


CPU = torch.device("cpu")


@pytest.fixture
def standalone_checkpoint(tmp_path):
    config = ExperimentConfig(
        time_proj_dim=8, class_proj_dim=8, unet_base_channels=4,
        unet_channel_mult=(1,), unet_num_res_blocks=1, unet_norm_groups=4,
        mixed_precision="none", num_workers=0,
    )
    model = build_model("pfm_delta", config)
    rng = capture_rng_state()
    rng["torch_cuda"] = None
    payload = build_training_checkpoint(
        model=model, optimizer=torch.optim.AdamW(model.parameters()),
        ema_model=model, config=config, variant="pfm_delta", epoch=3, step=12,
        loss=0.25, distributed_world_size=1, rng_state_by_rank=[rng],
        dataset_scientific_identity_sha256="d" * 64,
        runtime_identity={"project_name": "pfm", "git_commit": "a" * 40,
                          "rank": 0, "world_size": 1},
    )
    path = tmp_path / "downloaded-model.pt"
    torch.save(payload, path)
    return path, payload, config


def _sample_config():
    return SampleVisualizationConfig(
        profile="sample_visualization", sampling=SamplingConfig(steps=2),
        batch_size=1, class_start=3, class_count=2, num_per_class=1,
        seed=17, uncertainty_capture_t=0.5,
    )


def _evaluation_config():
    return PosteriorDiagnosticsEvaluationConfig(
        profile="posterior_diagnostics",
        metric_names=("posterior_controlled", "posterior_calibration"),
        posterior_population=2, batch_size=2, num_bins=2,
        posterior_t_values=(0.5,), seed=17,
    )


def test_standalone_binding_needs_no_run_tree_and_preserves_payload_identity(standalone_checkpoint):
    path, payload, config = standalone_checkpoint
    original = path.read_bytes()
    bound = checkpoint_loading.bind_checkpoint_file(path)
    loaded = checkpoint_loading.load_bound_training_checkpoint(bound)

    assert bound.checkpoint_path == path.resolve()
    assert bound.manifest_path is None
    assert bound.epoch_id == "epoch_0003"
    assert bound.manifest == {key: payload[key] for key in ("epoch", "step", "loss")}
    assert bound.checkpoint_sha256 == hashlib.sha256(original).hexdigest()
    assert bound.config_identity["variant"] == "pfm_delta"
    assert bound.config_identity["shared_training_protocol_sha256"] == shared_training_protocol_sha256(config)
    assert bound.checkpoint_runtime_identity == payload["runtime_identity"]
    for name, value in payload["ema_model_state"].items():
        assert torch.equal(loaded["ema_model_state"][name], value)
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]


def test_binding_hashes_the_bytes_loaded_even_when_path_is_atomically_replaced(
    standalone_checkpoint, monkeypatch,
):
    path, payload, _ = standalone_checkpoint
    original_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    replacement = path.with_name("replacement.pt")
    torch.save(dict(payload, epoch=4), replacement)
    original_load = torch.load

    def replace_after_load(handle, *args, **kwargs):
        selected = original_load(handle, *args, **kwargs)
        replacement.replace(path)
        return selected

    monkeypatch.setattr(checkpoint_io.torch, "load", replace_after_load)
    bound = checkpoint_loading.bind_checkpoint_file(path)
    monkeypatch.setattr(checkpoint_io.torch, "load", original_load)

    assert bound.manifest["epoch"] == 3
    assert bound.checkpoint_sha256 == original_digest
    assert hashlib.sha256(path.read_bytes()).hexdigest() != original_digest
    with pytest.raises(ValueError, match="digest does not match"):
        checkpoint_loading.load_bound_training_checkpoint(bound)


def test_standalone_sampling_needs_no_dataset_and_publishes_real_samples(
    standalone_checkpoint, tmp_path, monkeypatch,
):
    path, payload, _ = standalone_checkpoint
    def fail_dataset_access(*args, **kwargs):
        pytest.fail("standalone sampling must not access a training dataset")
    monkeypatch.setattr(checkpoint_loading, "validate_dataset_manifest", fail_dataset_access)
    monkeypatch.setattr(datasets, "make_datasets", fail_dataset_access)
    output = tmp_path / "samples"
    config = _sample_config()

    assert sample_grid.sample_checkpoint_file(
        path, config, output_dir=output, device=CPU, consumer_revision="b" * 40,
    ) == output

    result = json.loads((output / "result.json").read_text())
    provenance = json.loads((output / "checkpoint.json").read_text())
    assert result["population"] == 2
    assert result["sampling"]["protocol"] == asdict(config.sampling)
    assert result["sampling"]["uncertainty_capture_observed_t"] == 0.5
    page = result["presentation"]["pages"][0]
    assert (page["class_min"], page["class_max"]) == (3, 4)
    for name in ("samples", "velocity_variance", "endpoint_variance"):
        assert (output / page[name]).is_file()
    assert provenance["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance["producer"] == payload["runtime_identity"]
    assert provenance["consumer_revision"] == "b" * 40
    assert provenance["epoch"] == 3 and provenance["step"] == 12
    assert json.loads((output / "config.json").read_text()) == asdict(config)

    before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    monkeypatch.setattr(sample_grid, "iter_loaded_generation", lambda *a, **k:
                        pytest.fail("duplicate output must fail before generation"))
    with pytest.raises(FileExistsError, match="already exists"):
        sample_grid.sample_checkpoint_file(path, config, output_dir=output, device=CPU)
    assert {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("changes,match", [
    ({"class_start": 9}, "exceeds model classes"),
    ({"num_per_class": 501}, "limited to 1000"),
])
def test_standalone_sampling_keeps_class_and_population_bounds(
    standalone_checkpoint, tmp_path, changes, match,
):
    path, _, _ = standalone_checkpoint
    output = tmp_path / "samples"
    with pytest.raises(ValueError, match=match):
        sample_grid.sample_checkpoint_file(path, replace(_sample_config(), **changes),
                                           output_dir=output, device=CPU)
    assert not output.exists()


def test_standalone_evaluation_validates_dataset_and_calls_shared_metric_service(
    standalone_checkpoint, tmp_path, monkeypatch, execution_context,
):
    path, payload, _ = standalone_checkpoint
    config = _evaluation_config()
    output = tmp_path / "metrics"
    calls = []
    dataset_calls = []
    def dataset_manifest(root, dataset):
        dataset_calls.append((root, dataset))
        return {"scientific_identity_sha256": "d" * 64}
    def metrics(bound, staging, runtime, evaluation_config, device):
        calls.append((bound, runtime, evaluation_config, device))
        assert staging.parent == output.parent and not output.exists()
        loaded = checkpoint_loading.load_bound_training_checkpoint(bound)
        assert loaded["epoch"] == 3
        (staging / "metric-detail.json").write_text('{"checked":true}')
        return {"profile": config.profile, "method": "pfm_delta", "checked": True}
    monkeypatch.setattr(checkpoint_loading, "validate_dataset_manifest", dataset_manifest)
    monkeypatch.setattr(checkpoint_evaluation, "evaluate_checkpoint_metrics", metrics)

    result = checkpoint_evaluation.evaluate_checkpoint_file(
        path, config, output_dir=output, job_context=execution_context, device=CPU,
    )

    assert result == {"artifact_dir": str(output), "result": str(output / "result.json")}
    assert dataset_calls == [(execution_context.runtime.dataset_root("mnist"), "mnist")]
    assert len(calls) == 1
    bound, runtime, selected_config, device = calls[0]
    assert (runtime, selected_config, device) == (execution_context.runtime, config, CPU)
    assert bound.manifest_path is None
    assert json.loads((output / "result.json").read_text())["checked"] is True
    assert (output / "metric-detail.json").is_file()
    provenance = json.loads((output / "checkpoint.json").read_text())
    assert provenance["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance["producer"] == payload["runtime_identity"]
    assert provenance["consumer_revision"] == execution_context.git_commit
    with pytest.raises(FileExistsError, match="already exists"):
        checkpoint_evaluation.evaluate_checkpoint_file(
            path, config, output_dir=output, job_context=execution_context, device=CPU,
        )
    assert len(calls) == 1


def test_standalone_evaluation_rejects_dataset_mismatch_before_metrics_or_output(
    standalone_checkpoint, tmp_path, monkeypatch, execution_context,
):
    path, _, _ = standalone_checkpoint
    output = tmp_path / "metrics"
    monkeypatch.setattr(checkpoint_loading, "validate_dataset_manifest", lambda *args:
                        {"scientific_identity_sha256": "e" * 64})
    monkeypatch.setattr(checkpoint_evaluation, "evaluate_checkpoint_metrics", lambda *args:
                        pytest.fail("mismatched dataset must fail before metrics"))
    with pytest.raises(ValueError, match="dataset scientific identity mismatch"):
        checkpoint_evaluation.evaluate_checkpoint_file(
            path, _evaluation_config(), output_dir=output,
            job_context=execution_context, device=CPU,
        )
    assert not output.exists()


@pytest.mark.parametrize("consumer", ["sample", "evaluate"])
def test_standalone_consumer_removes_incomplete_output_after_computation_failure(
    standalone_checkpoint, tmp_path, monkeypatch, execution_context, consumer,
):
    path, _, _ = standalone_checkpoint
    output = tmp_path / "incomplete"
    def fail(*args, **kwargs):
        raise RuntimeError("test computation failure")
    monkeypatch.setattr(checkpoint_loading, "validate_dataset_manifest", lambda *args:
                        {"scientific_identity_sha256": "d" * 64})
    if consumer == "sample":
        monkeypatch.setattr(sample_grid, "iter_loaded_generation", fail)
        consume = lambda: sample_grid.sample_checkpoint_file(
            path, _sample_config(), output_dir=output, device=CPU)
    else:
        monkeypatch.setattr(checkpoint_evaluation, "evaluate_checkpoint_metrics", fail)
        consume = lambda: checkpoint_evaluation.evaluate_checkpoint_file(
            path, _evaluation_config(), output_dir=output,
            job_context=execution_context, device=CPU)
    with pytest.raises(RuntimeError, match="test computation failure"):
        consume()
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.staging-*"))
